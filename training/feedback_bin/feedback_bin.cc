/*
Author: Avneesh Saluja (avneesh@cs.cmu.edu)
based off of latent_svm.cc code by Jeff Flanigan (jflanigan@cs.cmu.edu) and Avneesh Saluja
latent_svm.cc code based off kbest_mira.cc code by Chris Dyer (cdyer@cs.cmu.edu)

Points to note regarding variable names:
total_loss and prev_loss actually refer not to loss, but the metric (usually BLEU)
 */
#include <sstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

//boost libraries
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

//cdec libraries
#include "config.h"
#include "hg_sampler.h"
#include "sentence_metadata.h"
#include "scorer.h"
#include "verbose.h"
#include "viterbi.h"
#include "hg.h"
#include "prob.h"
#include "kbest.h"
#include "ff_register.h"
#include "decoder.h"
#include "filelib.h"
#include "fdict.h"
#include "weights.h"
#include "sparse_vector.h"
#include "sampler.h"
#include "stringlib.h" //added to count # of tokens in source (normalizer)

using namespace std;
using boost::shared_ptr;
namespace po = boost::program_options;

bool invert_score; 
boost::shared_ptr<MT19937> rng; //random seed ptr

void RandomPermutation(int len, vector<int>* p_ids) {
  vector<int>& ids = *p_ids;
  ids.resize(len);
  for (int i = 0; i < len; ++i) ids[i] = i;
  for (int i = len; i > 0; --i) {
    int j = rng->next() * i;
    if (j == i) i--;
    swap(ids[i-1], ids[j]);
  }  
}

bool InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("weights,w",po::value<string>(),"[REQD] Input feature weights file")
        ("source,i",po::value<string>(),"[REQD] Input source file for development (supervised) set")
        ("posSource,P",po::value<string>(),"positive labeled source sentences file") 
        ("negSource,N",po::value<string>(),"negative labeled source sentences file")
        ("passes,p", po::value<int>()->default_value(15), "Number of passes through the training data")
        ("reference,r",po::value<vector<string> >(), "[REQD] Reference translation(s) (tokenized text file) for development (supervised) set")
        ("pos_ref,R",po::value<vector<string> >(),"positive labeled 'reference' (i.e., 1-best hypothesis) sentences file")
        ("neg_ref,V",po::value<vector<string> >(),"negative labeled 'reference' (i.e., 1-best hypothesis) sentences file") 
        ("jlis_pos,q", "turn on J-LIS positive examples training")
        ("jlis_neg,j", "turn on J-LIS negative examples training")
        ("output_weights,o", "output weights after going through each sub-corpus (sup, pos, neg) in a pass")
        ("batch_online,B", "choice of batch or online gradient descent.  Batch updates after every sub-pass.  Default is online/stochastic")
        ("mt_metric,m",po::value<string>()->default_value("ibm_bleu"), "Scoring metric (ibm_bleu, nist_bleu, koehn_bleu, ter, combi)")
        ("regularizer_strength,C", po::value<double>()->default_value(0.01), "regularization strength")
        ("mt_metric_scale,s", po::value<double>()->default_value(1.0), "Cost function is -mt_metric_scale*BLEU")
        ("costaug_log_bleu,l", "Cost function is -mt_metric_scale*log(BLEU)")
        ("source_length_norm_off,L", "Turn off source length normalization for features values")
        ("mu,u", po::value<double>()->default_value(0.0), "weight (between 0 and 1) to scale model score by for oracle selection")
        ("stepsize_param,a", po::value<double>()->default_value(0.01), "Stepsize parameter during supervised (L-SSVM) optimization")
        ("stepsize_jlis,v", po::value<double>()->default_value(0.01), "Stepsize parameter during binary feedback (JLIS) optimization")
        ("stepsize_reduce,t", "Divide step size by sqrt(number of examples seen so far), as per Ratliff et al, 2007")
        ("fixweight,f",po::value<string>(), "fix weight of feature provided as string argument")
        ("featureFilePos,F", po::value<string>(), "instead of redecoding input sentences, provide feature values for each chunked sentence which are then used to update the weights; positive sentences")
        ("featureFileNeg,T", po::value<string>(), "instead of redecoding input sentences, provide feature values for each chunked sentence which are then used to update the weights; positive sentences")
        ("k_best_size,k", po::value<int>()->default_value(250), "Size of hypothesis list to search for oracles")
        ("best_ever,b", "Keep track of the best hypothesis we've ever seen (metric score), and use that as the reference")
        ("random_seed,S", po::value<uint32_t>(), "Random seed (if not specified, /dev/random will be used)")
        ("decoder_config,c",po::value<string>(),"Decoder configuration file");
  po::options_description clo("Command line options");
  clo.add_options()
        ("config", po::value<string>(), "Configuration file")
        ("help,h", "Print this help message and exit");
  po::options_description dconfig_options, dcmdline_options;
  dconfig_options.add(opts);
  dcmdline_options.add(opts).add(clo);
  
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("config")) {
    ifstream config((*conf)["config"].as<string>().c_str());
    po::store(po::parse_config_file(config, dconfig_options), *conf);
  }
  po::notify(*conf);

  if (conf->count("help") || !conf->count("weights") || !conf->count("source") || !conf->count("decoder_config") || !conf->count("reference")) {
    cerr << dcmdline_options << endl;
    return false;
  }
  return true;
}

double cost_augmented_score(const LogVal<double> model_score, const double mt_metric_score, const double mt_metric_scale, const bool logbleu) {
  if(logbleu) {
    if(mt_metric_score != 0)
      // NOTE: log(model_score) is just the model score feature weights * features
      return log(model_score) + (-mt_metric_scale * log(mt_metric_score)); 
    else
      return -1000000;
  }
  // NOTE: log(model_score) is just the model score feature weights * features
  return log(model_score) + (- mt_metric_scale * mt_metric_score);
}

double muscore(const vector<weight_t>& feature_weights, const SparseVector<double>& feature_values, const double mt_metric_score, const double mu, const bool logbleu) {
  if(logbleu) {
    if(mt_metric_score != 0)
      return feature_values.dot(feature_weights) * mu + (1 - mu) * log(mt_metric_score);
    else
      return feature_values.dot(feature_weights) * mu + (1 - mu) * (-1000000);  // log(0) is -inf
  }
  return feature_values.dot(feature_weights) * mu + (1 - mu) * mt_metric_score;
}

static const double kMINUS_EPSILON = -1e-6;

struct HypothesisInfo {
  SparseVector<double> features;
  double mt_metric_score;
  // The score changes when the feature weights change, so it is not stored here
  // It must be recomputed every time
};

struct GoodOracle {
  shared_ptr<HypothesisInfo> good;
};

struct TrainingObserver : public DecoderObserver {
  TrainingObserver(const int k,
                   const DocScorer& d,
                   vector<GoodOracle>* o,
                   const vector<weight_t>& feat_weights,
                   const double metric_scale,
                   const double Mu,
                   const bool bestever,
                   const bool LogBleu) : ds(d), oracles(*o), feature_weights(feat_weights), kbest_size(k), mt_metric_scale(metric_scale), best_ever(bestever), mu(Mu), log_bleu(LogBleu) {}
  const DocScorer& ds;
  const vector<weight_t>& feature_weights;
  vector<GoodOracle>& oracles;
  shared_ptr<HypothesisInfo> cur_best;
  shared_ptr<HypothesisInfo> cur_costaug_best;
  const int kbest_size;
  const double mt_metric_scale;
  const double mu;
  const bool best_ever;
  const bool log_bleu;

  HypothesisInfo& GetCurrentBestHypothesis() {
    return *cur_best;
  }

  const HypothesisInfo& GetCurrentCostAugmentedHypothesis() const {
    return *cur_costaug_best;
  }

  virtual void NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg) {
    UpdateOracles(smeta.GetSentenceID(), *hg);
  }

  shared_ptr<HypothesisInfo> MakeHypothesisInfo(const SparseVector<double>& feats, const double metric) {
    shared_ptr<HypothesisInfo> h(new HypothesisInfo);
    h->features = feats;
    h->mt_metric_score = metric;
    return h;
  }

  void UpdateOracles(int sent_id, const Hypergraph& forest) {
    shared_ptr<HypothesisInfo>& cur_ref = oracles[sent_id].good;
    if(!best_ever)
      cur_ref.reset();
    KBest::KBestDerivations<vector<WordID>, ESentenceTraversal> kbest(forest, kbest_size);
    double costaug_best_score = 0;

    for (int i = 0; i < kbest_size; ++i) {
      const KBest::KBestDerivations<vector<WordID>, ESentenceTraversal>::Derivation* d =
        kbest.LazyKthBest(forest.nodes_.size() - 1, i);
      if (!d) break;
      double mt_metric_score = ds[sent_id]->ScoreCandidate(d->yield)->ComputeScore(); //this might need to change!
      const SparseVector<double>& feature_vals = d->feature_values;
      double costaugmented_score = cost_augmented_score(d->score, mt_metric_score, mt_metric_scale, log_bleu); //note that d->score i.e., model score is passed in
      if (i == 0) { //top entry in the kbest list, setting up our cur_best to be model score highest, and initializing costaug_best
        cur_best = MakeHypothesisInfo(feature_vals, mt_metric_score);
        cur_costaug_best = cur_best;
        costaug_best_score = costaugmented_score; 
      }
      if (costaugmented_score > costaug_best_score) {   // "fear' derivation
        cur_costaug_best = MakeHypothesisInfo(feature_vals, mt_metric_score);
        costaug_best_score = costaugmented_score;
      }
      double cur_muscore = mt_metric_score;
      if (!cur_ref)   // "hope" derivation
        cur_ref =  MakeHypothesisInfo(feature_vals, cur_muscore);
      else {
          double cur_ref_muscore = cur_ref->mt_metric_score;
          if(mu > 0) { //select oracle with mixture of model score and BLEU
              cur_ref_muscore =  muscore(feature_weights, cur_ref->features, cur_ref->mt_metric_score, mu, log_bleu);
              cur_muscore = muscore(feature_weights, d->feature_values, mt_metric_score, mu, log_bleu);
          }
          if (cur_muscore > cur_ref_muscore) //replace oracle
            cur_ref = MakeHypothesisInfo(feature_vals, mt_metric_score);
      }
    }
  }
};

//generic function to split a string given a delimiter
vector<string> splitStr (const string& str, const char& delim){
  typedef string::const_iterator iter; 
  iter beg = str.begin();
  vector<string> tokens; 
  while (beg != str.end()){
    iter temp = find(beg, str.end(), delim); 
    if (beg != str.end()){ tokens.push_back(string(beg, temp)); } //if beg is not at the end, add the token to tokens
    beg = temp; //set beg to be where the iterator found the delimiter
    while ((beg != str.end()) && (*beg == delim)){ beg++; } //if beg is not at the end and is the delimiter, increment
  }
  return tokens; 
}

void ReadTrainingCorpus(const string& fname, vector<string>* c) {
  ReadFile rf(fname);
  istream& in = *rf.stream();
  string line;
  while(in) {
    getline(in, line);
    if (!in) break;
    c->push_back(line);
  }
}

//experimental code for handling longer sentences using parse trees
void ReadFeatureFile(const string& fname, vector<SparseVector<double> >* feats){
  ReadFile rf(fname); 
  istream& in = *rf.stream();
  string line; 
  while(in) {
    getline(in, line); 
    vector<double> vec; 
    vector<string> tokens = splitStr(line, ' '); 
    for (unsigned int i = 0; i < tokens.size(); i++ ){
      vector<string> token = splitStr(tokens[i], '='); 
      const int fid = FD::Convert(token[0]); //converting feature name to fid
      double val = strtod(token[1].c_str(), NULL); 
      if (isnan(val)){
	cerr << FD::Convert(fid) << "has feature value NaN!\n"; 
	abort(); 
      }
      if (vec.size() <= fid){ vec.resize(fid+1); }
      vec[fid] = val;     
    }        
    SparseVector<double> sparseVec; 
    sparseVec.clear(); 
    for (unsigned int i = 0; i < vec.size(); i++ ){
      if (vec[i]) sparseVec.set_value(i, vec[i]);
    }  
    feats->push_back(sparseVec); //at this stage, vec has all the feature values for a particular line
  }
}

bool ApproxEqual(double a, double b) {
  if (a == b) return true;
  return (fabs(a-b)/fabs(b)) < 0.000001;
}

int main(int argc, char** argv) {
  register_feature_functions();
  SetSilent(true);  // turn off verbose decoder output

  po::variables_map conf;
  if (!InitCommandLine(argc, argv, &conf)) return 1;

  if (conf.count("random_seed")) //random seed init
    rng.reset(new MT19937(conf["random_seed"].as<uint32_t>()));
  else
    rng.reset(new MT19937);

  const bool best_ever = conf.count("best_ever") > 0; 
  vector<string> corpus, posSource, negSource; //supervised, and binary (positive and negative) source corpora
  ReadTrainingCorpus(conf["source"].as<string>(), &corpus);  
  if (conf.count("posSource") && conf.count("negSource")){ //if +/- corpora defined, we read them in
    ReadTrainingCorpus(conf["posSource"].as<string>(), &posSource); 
    ReadTrainingCorpus(conf["negSource"].as<string>(), &negSource); 
  }

  //experimental code for feature handling from synchronous parse tree
  vector<SparseVector<double> > posFeatures, negFeatures; 
  if (conf.count("featureFilePos") && conf.count("featureFileNeg")){
    ReadFeatureFile(conf["featureFilePos"].as<string>(), &posFeatures); 
    ReadFeatureFile(conf["featureFileNeg"].as<string>(), &negFeatures); 
  }

  //code to handle fixing feature weight updating for particular features
  string fixfeature; 
  const bool fixweight = conf.count("fixweight") > 0;  
  if (fixweight){
    fixfeature = conf["fixweight"].as<string>(); 
  }

  const string metric_name = conf["mt_metric"].as<string>(); //set up metric name
  ScoreType type = ScoreTypeFromString(metric_name);
  if (type == TER) {
    invert_score = true;
  } else {
    invert_score = false;
  }

  DocScorer ds(type, conf["reference"].as<vector<string> >(), ""); //read in ref
  cerr << "Loaded " << ds.size() << " references for scoring with " << metric_name << endl;
  if (ds.size() != corpus.size()) {
    cerr << "Mismatched number of references (" << ds.size() << ") and sources (" << corpus.size() << ")\n";
    return 1;
  }

  //in current version of code, we're required to read in posRef and negRef; change scoping to alter this behavior
  DocScorer posRef(type, conf["pos_ref"].as<vector<string> >(), ""); //read in posRef   
  cerr << "Loaded " << posRef.size() << " positive reference files" << endl;    
  DocScorer negRef(type, conf["neg_ref"].as<vector<string> >(), ""); //read in negRef
  cerr << "Loaded " << negRef.size() << " negative reference files" << endl; 
  if (posRef.size() != posSource.size() || negRef.size() != negSource.size()){ 
    cerr << "Mismatched number of references (pos, neg) = (" << posRef.size() << "," << negRef.size() << ") and sources (pos, neg) = (" << posSource.size() << "," << negSource.size() << ")\n"; 
    return 1;
  }
 
  ReadFile ini_rf(conf["decoder_config"].as<string>()); //read in decoder config file
  Decoder decoder(ini_rf.stream()); //init decoder object

  // load initial weights
  vector<weight_t>& decoder_weights = decoder.CurrentWeightVector(); //setting up the pointer for the decoder current weights
  SparseVector<weight_t> sparse_weights; 
  Weights::InitFromFile(conf["weights"].as<string>(), &decoder_weights); //read in input weights to decoder weights
  //uncomment below code useful for debugging purposes
  //Weights::ShowLargestFeatures(decoder_weights);
  Weights::InitSparseVector(decoder_weights, &sparse_weights); //transfer decoder weights to sparse weights

  //initializing other algorithm/output parameters
  const double c = conf["regularizer_strength"].as<double>();
  const double mt_metric_scale = conf["mt_metric_scale"].as<double>();
  const double mu = conf["mu"].as<double>();
  const double stepsize_param = conf["stepsize_param"].as<double>(); //step size in structured SGD optimization step
  const double stepsize_jlis = conf["stepsize_jlis"].as<double>(); //ditto
  const bool stepsize_reduce = conf.count("stepsize_reduce") > 0; 
  const bool costaug_log_bleu = conf.count("costaug_log_bleu") > 0;
  const bool src_len_norm_off = conf.count("source_length_norm_off") > 0; 
  const bool jlis_pos = conf.count("jlis_pos") > 0; 
  const bool jlis_neg = conf.count("jlis_neg") > 0; 
  const bool output_weights = conf.count("output_weights") > 0; 
  const bool batch = conf.count("batch_online") > 0; 

  assert(corpus.size() > 0);
  vector<GoodOracle> oracles(corpus.size());
  vector<GoodOracle> posOracles(posSource.size()); 
  vector <GoodOracle> negOracles(negSource.size()); 
  TrainingObserver observer(conf["k_best_size"].as<int>(),  // kbest size 
                            ds,                             // doc scorer
                            &oracles,
                            decoder_weights,
                            mt_metric_scale,
                            mu,
                            best_ever,
                            costaug_log_bleu);
  //need to declare "observers" for +/- corpora
  TrainingObserver posObserver(conf["k_best_size"].as<int>(),
			         posRef,
			         &posOracles, 
				 decoder_weights,
				 mt_metric_scale,
				 mu,
				 best_ever,
				 costaug_log_bleu); 
  TrainingObserver negObserver(conf["k_best_size"].as<int>(),
				 negRef,
			         &negOracles, 
				 decoder_weights,
				 mt_metric_scale,
				 mu,
				 best_ever,
				 costaug_log_bleu); 
			       
  int normalizer = 0; 
  double total_loss = 0;
  double prev_loss = 0;
  int dots = 0;             // progess bar
  SparseVector<double> tot; 
  tot += sparse_weights; 
  normalizer++; 
  int max_iteration = conf["passes"].as<int>();
  int cur_pass = 0;
  string msg = "# LatentSVM tuned weights";
  vector<int> order, posOrder, negOrder;
  int interval_counter = 0;  
  while (cur_pass < max_iteration){
    RandomPermutation(corpus.size(), &order);
    RandomPermutation(posSource.size(), &posOrder);
    RandomPermutation(negSource.size(), &negOrder); 
    cerr << "PASS " << cur_pass << endl; 
    int cur_sent = 0;     
    while (cur_sent < corpus.size()){
      if (( interval_counter * 40 / corpus.size()) > dots ) {++dots; cerr << ".";}
      if (!batch){
	sparse_weights.init_vector(&decoder_weights);   //only if we're updating online should we update our decoder weights
      }
      decoder.SetId(order[cur_sent]);
      decoder.Decode(corpus[order[cur_sent]], &observer);  // update oracles

      //const HypothesisInfo& cur_best = observer.GetCurrentBestHypothesis(); //model score best
      HypothesisInfo& cur_best = observer.GetCurrentBestHypothesis(); //non-const version
      const HypothesisInfo& cur_costaug = observer.GetCurrentCostAugmentedHypothesis(); //model + bleu best; cost augmented best; uses costaug_score function
      const HypothesisInfo& cur_ref = *oracles[order[cur_sent]].good; //does not use mu_score function
      //const HypothesisInfo& cur_ref = observer.GetCurrentReference(); //if mu > 0, this mu-mixed oracle will be picked, otherwise only on BLEU
      total_loss += cur_best.mt_metric_score; //A: added this; is this correct? 
      const double loss = cur_costaug.features.dot(decoder_weights) - cur_ref.features.dot(decoder_weights);
      // w_{t+1} = w_t - stepsize_t * grad(Loss) where stepsize_t = alpha/t    
      double step_size = stepsize_param;
      if (stepsize_reduce){
	step_size  /= (sqrt(cur_sent+1.0)); 
      }
      //update the weights
      sparse_weights -= cur_costaug.features * step_size; // cost aug hyp orig -
      sparse_weights+= cur_ref.features * step_size; //ref orig +
      sparse_weights -= sparse_weights * (c * step_size); //regularizer term
      tot += sparse_weights; 
      normalizer++; 
      interval_counter++;
      cur_sent++;
    }    
    cerr << "Finished supervised corpus\n";
    cerr << " [AVG METRIC LAST PASS=" << (total_loss / corpus.size()) << "]\n";
    cerr << " TOTAL LOSS: " << total_loss << "\n";
    sparse_weights.init_vector(&decoder_weights);   // copy sparse_weights to the decoder weights
    Weights::ShowLargestFeatures(decoder_weights);
    cur_sent = 0;
    total_loss = 0;
    SparseVector<double> x = tot; 
    x /= normalizer; 
    x.init_vector(&decoder_weights); //copy normalized weights to decoder weights
    dots = 0;
    if (output_weights){ //if the appropriate command line option is set, we output 
      ostringstream os;
      os << "weights.latentsvm-sup-pass" << (cur_pass < 10 ? "0" : "") << cur_pass << ".gz";
      Weights::WriteToFile(os.str(), decoder_weights, true, &msg);
    }

    //loop through +ve corpus now
    if (jlis_pos){
      cerr << "Starting positive corpus\n"; 
    while (cur_sent < posSource.size()){
	if (( interval_counter * 40 / posSource.size()) > dots ) {++dots; cerr << ".";}
	SparseVector<double> curFeat; 
	if (conf.count("featureFilePos")){
	  curFeat = posFeatures[posOrder[cur_sent]]; //posFeatures holds a vector of sparse vectors
	}
	else {	
	  decoder.SetId(posOrder[cur_sent]);
	  decoder.Decode(posSource[posOrder[cur_sent]], &posObserver);  // update oracles              
	  HypothesisInfo& cur_best = posObserver.GetCurrentBestHypothesis(); 
	  curFeat = cur_best.features; 
	}
	unsigned srclen = NTokens(posSource[posOrder[cur_sent]], ' '); 
	if (srclen > 0){ //ignore the example if = 0
	  const double loss = 1-curFeat.dot(decoder_weights)/srclen; 
	  cout << "Loss is " << loss << endl; 
	  total_loss += loss; 
	  double step_size = stepsize_jlis;
	  if (fixweight){ 	//zeroing out LM feature
	    curFeat.set_value(FD::Convert(fixfeature), 0.0); 
	  }
	  if (loss > 0.0){
	    if (stepsize_reduce){
	      step_size  /= (sqrt(cur_sent+1.0));
	    }
	    if (src_len_norm_off){
	      srclen = 1; 
	    }
	    sparse_weights += (curFeat*step_size)/srclen; 
	  }
	  if (!batch){
	    sparse_weights.init_vector(&decoder_weights); 
	  }
	  tot += sparse_weights; 
	  normalizer++; 
	  interval_counter++; 
	}
	cur_sent++; 	
    }
    cerr << "Finished positive corpus\n"; 
    cerr << " [AVG METRIC LAST PASS=" << ( total_loss / posSource.size()) << "]\n"; 
    cerr << " TOTAL LOSS: " << total_loss << "\n"; 
    Weights::ShowLargestFeatures(decoder_weights); 
    cur_sent=0; 
    total_loss = 0; 
    interval_counter = 0; 
    dots = 0; 
    x = tot; 
    x /= normalizer; 
    x.init_vector(&decoder_weights); 
    if (output_weights){ //if the appropriate command line option is set, we output 
	ostringstream os;
	os << "weights.latentsvm-pos-pass" << (cur_pass < 10 ? "0" : "") << cur_pass << ".gz";
	Weights::WriteToFile(os.str(), decoder_weights, true, &msg);
      }
    }
    if (jlis_neg) {
      cerr << "Starting negative corpus\n"; 
      while (cur_sent < negSource.size()){
	if (( interval_counter * 40 / negSource.size()) > dots ) {++dots; cerr << ".";}
	SparseVector<double> curFeat; 
	if (conf.count("featureFileNeg")){
	  curFeat = negFeatures[negOrder[cur_sent]]; 
	}
	else {
	  decoder.SetId(negOrder[cur_sent]);
	  decoder.Decode(negSource[negOrder[cur_sent]], &negObserver); //update oracles
	  HypothesisInfo& cur_best = negObserver.GetCurrentBestHypothesis(); 
	  curFeat = cur_best.features; 
	}
	unsigned srclen = NTokens(negSource[negOrder[cur_sent]], ' '); 
	if (srclen > 0){
	  const double loss = 1 + curFeat.dot(decoder_weights)/srclen; //dividing by sentence length as per J-LIS paper
	  total_loss += loss; 
	  double step_size = stepsize_jlis; 
	  if (fixweight) {
	    curFeat.set_value(FD::Convert(fixfeature), 0.0); 
	  }
	  if (loss > 0.0){ //should always be the case since we're adding? 
	    if (stepsize_reduce) {
	      step_size /= (sqrt(cur_sent+1.0)); 
	    }
	    if (src_len_norm_off){
	      srclen = 1; 
	    }
	    sparse_weights -= (curFeat*step_size)/srclen; //update sparse weights
	  }      
	  if (!batch){
	    sparse_weights.init_vector(&decoder_weights); 
	  }	
	  tot += sparse_weights; 
	  normalizer++; 
	  interval_counter++; 
	}
	cur_sent++;	
      }
      cerr << "Finished negative corpus\n";
      cerr << " [AVG METRIC LAST PASS=" << ( total_loss / negSource.size()) << "]\n";
      cerr << " TOTAL LOSS: " << total_loss << "\n";
      Weights::ShowLargestFeatures(decoder_weights); 
      cur_sent=0;
      total_loss = 0; 
      x = tot; 
      x /= normalizer;
      x.init_vector(&decoder_weights);
      dots = 0; 
      interval_counter = 0; 
      if (output_weights){ //if the appropriate command line option is set, we output 
	ostringstream os;
	os << "weights.latentsvm-neg-pass" << (cur_pass < 10 ? "0" : "") << cur_pass << ".gz";
	Weights::WriteToFile(os.str(), decoder_weights, true, &msg);
      }
    } //if conf.count posSource negSource or the jlis condition
    cerr << "Normalized weights: \n"; 
    Weights::ShowLargestFeatures(decoder_weights); 
    ostringstream os;
    os << "weights.latentsvm-pass" << (cur_pass < 10 ? "0" : "") << cur_pass << ".gz";
    Weights::WriteToFile(os.str(), decoder_weights, true, &msg);
    cur_pass++;
  } 
  //completed training; now write to final weights file
  cerr << endl;
  Weights::WriteToFile("weights.latentsvm-final.gz", decoder_weights, true, &msg);
  tot /= normalizer;
  tot.init_vector(decoder_weights); 
  msg = "# Latent SSVM tuned weights (averaged vector)";
  Weights::WriteToFile("weights.latentsvm-final-avg.gz", decoder_weights, true, &msg); 
  cerr << "Optimization complete.\n" ; 
  return 0;
}

