bin_PROGRAMS = feedback_bin

feedback_bin_SOURCES = feedback_bin.cc
feedback_bin_LDADD = ../..//decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a

AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(top_srcdir)/utils -I$(top_srcdir)/decoder -I$(top_srcdir)/mteval
