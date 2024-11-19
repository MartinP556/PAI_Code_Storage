#!/bin/ksh
#
#
###################################################################################
#
#
###################################################################################
# Module's directory structure
#==================================================================================
#DIR_ROOT=$(cd $(dirname $0) && pwd) || exit 1                #    module root path
DIR_ROOT=$1/modules/                                          #    module root path LS
#----------------------------------------------------------------------------------
DIR_COMMON=${DIR_ROOT}/common                                 # common modules path
DIR_CYCLE=${DIR_ROOT}/cycle                                   #  cycle modules path
DIR_CORE=${DIR_ROOT}/core                                     #   core modules path
DIR_MORE=${DIR_ROOT}/more                                     #   more modules path
DIR_MEC=${DIR_ROOT}/mec                                       #    mec modules path
DIR_INT2LM=${DIR_ROOT}/int2lm                                 # int2lm modules path
DIR_SNOW=${DIR_ROOT}/snow                                     #   snow modules path
DIR_SMA=${DIR_ROOT}/sma                                       #    sma modules path
DIR_SST=${DIR_ROOT}/sst                                       #    sst modules path
#----------------------------------------------------------------------------------
DEBUG=${G_DEBUG:-100}                                         # debug info level
###################################################################################
#
#
###################################################################################
# Initialization (general part)
#==================================================================================
export G_TOPLEVEL=${G_TOPLEVEL:-$SHLVL}         # determination of top level script
#----------------------------------------------------------------------------------
source $DIR_COMMON/error_fcns.sh                               # for error handling
trap 'f_error_handler $LINENO ER_msg' ERR
f_error_push_stacks SCRIPT "$0" "${.sh.file}"
#----------------------------------------------------------------------------------
#export COMMON_VAR_IO_PATH=$DIR_ROOT                         # for common variables
export COMMON_VAR_IO_PATH=$3                                 # for common variables LS
export COMMON_VAR_IO_NAME=common_var.print_config
[ "$SHLVL" -eq "$G_TOPLEVEL" ]   && $DIR_COMMON/common_var_init.sh
#----------------------------------------------------------------------------------
source $DIR_COMMON/report_fcns.sh && f_report_init $DEBUG        # formatted output
#----------------------------------------------------------------------------------
source $DIR_COMMON/call_fcns.sh   && f_call_init                   # call functions
###################################################################################
#
#
###################################################################################
# Initialization (module specific part)
#==================================================================================
source $DIR_COMMON/bacy_fcns.sh && f_bacy_init             # bacy specific functions
###################################################################################
#
#
###################################################################################
# Usage
#==================================================================================
typeset usage=$(cat << EOF

Usage: ${0##*/} exp_path model tmp_path
         - 'exp_path': path of bacy experiment (containing modules directory)
         - 'model'   : NWP model; options: ICON|COSMO|ICON-LAM
         - 'tmp_path': temporary directory for common variables communication
         
EOF
)
###################################################################################
#
#
###################################################################################
# Checking/obtaining input options and arguments
#==================================================================================
if [ "$#" -ne 3 ]; then
  f_error_set_msg_and_send 1 ER_msg \
    "Wrong number ($#) of input arguments." \
    "$usage"
fi
# second input argument
case "$2" in
  ICON|ICON-LAM|COSMO)
    typeset in_model=$2
    ;;
  *)
    f_error_set_msg_and_send 1 ER_msg \
      "Invalid value ($2) for second input argument." \
      "$usage"
      ;;
esac
###################################################################################
#
#
###################################################################################
# Some greeting message
#==================================================================================
if [ "$SHLVL" -eq "$G_TOPLEVEL" ]; then
  f_report_star 4 "Showing BACY configurations for model $in_model"
fi
###################################################################################
#
#
###################################################################################
# Loading configurations
#==================================================================================
# BACY
f_report_star 3 "Showing bacy_conf.sh"
f_call_load_conf $DIR_COMMON/bacy_conf.sh $in_model
#----------------------------------------------------------------------------------
# CYCLE
f_report_star 3 "Showing cycle_conf.sh"
CY_IN_MODEL=$in_model
f_call_load_conf $DIR_CYCLE/cycle_conf.sh
#----------------------------------------------------------------------------------
# GET_DATA
f_report_star 3 "Showing get_data_conf.sh"
f_call_load_conf $DIR_COMMON/get_data_conf.sh
#----------------------------------------------------------------------------------
# CORE
f_report_star 3 "Showing core_conf.sh"
CO_IN_MODEL=$in_model
CO_IN_METHOD=$CY_METHOD
CO_OPT_ADJUST=0
f_call_load_conf $DIR_CORE/core_conf.sh
#----------------------------------------------------------------------------------
# MORE
f_report_star 3 "Showing more_conf.sh"
MO_COMMON=$DIR_COMMON
f_call_load_conf $DIR_MORE/more_conf.sh $in_model
#----------------------------------------------------------------------------------
# MEC
if [[ "in_model" = @(ICON-LAM|COSMO) ]]; then
  case "${BA_L_USE[*]}" in
    *[4-6]*)
      f_report_star 3 "Showing mec_conf.sh"
      ME_IN_MODEL=$in_model
      f_call_load_conf $DIR_MEC/mec_conf.sh
      ;;
  esac
fi
#----------------------------------------------------------------------------------
# INT2LM
if [[ "in_model" = @(ICON-LAM|COSMO) ]]; then
  f_report_star 3 "Showing int2lm_conf.sh"
  f_call_load_conf $DIR_INT2LM/int2lm_conf.sh $in_model
fi
#----------------------------------------------------------------------------------
# SNOW
SN_IN_MODEL=$in_model
if [ "$BA_RUN_SNW" -ne 0 ]; then
  f_report_star 3 "Showing snow_conf.sh"
  f_call_load_conf $DIR_SNOW/snow_conf.sh
fi
#----------------------------------------------------------------------------------
# SMA
if [ "$BA_RUN_SMA" -ne 0 ]; then
  f_report_star 3 "Showing sma_conf.sh"
  f_call_load_conf $DIR_SMA/sma_conf.sh
fi
#----------------------------------------------------------------------------------
# SST
ST_IN_MODEL=$in_model
if [ "$BA_RUN_SST" -ne 0 ]; then
  f_report_star 3 "Showing sst_conf.sh"
  f_call_load_conf $DIR_SST/sst_conf.sh
fi
###################################################################################
# End of script
#==================================================================================
f_report_star 4 "Finished"
#----------------------------------------------------------------------------------
rm $COMMON_VAR_IO_PATH/$COMMON_VAR_IO_NAME
exit 0
###################################################################################
