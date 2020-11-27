import numpy as np
import sys
import os
import logging
import datetime
from keras import optimizers, regularizers
from keras.models import load_model, Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Flatten, Reshape, TimeDistributed, Bidirectional, CuDNNLSTM, CuDNNGRU, GRU, \
    Dropout, Input, Conv2D, MaxPool2D, ConvLSTM2D, SpatialDropout2D, Conv1D, MaxPool1D, Concatenate, BatchNormalization, \
    Activation, AveragePooling2D, Embedding
from keras.utils import plot_model
import time
import argparse
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import class_weight
from predict import log_time, get_array_of_float_from_a_line, Read1DFeature, ReadBERTFeature, Split1Dlist2NpArrays, \
    Split2DList2NpArrays, Convert2DListTo3DNp


# load each feature dictionary
def LoadFeatures(args):
    log_time("Loading feature ECO")
    Read1DFeature(args.tmp_dir + "/ECO.txt", ECO_test_dic)
    log_time("Loading feature RAA")
    Read1DFeature(args.tmp_dir + "/RAA.txt", RAA_test_dic)
    log_time("Loading feature RSA")
    Read1DFeature(args.tmp_dir + "/RSA.txt", RSA_test_dic)
    log_time("Loading feature Anchor")
    Read1DFeature(args.tmp_dir + "/Anchor.txt", Anchor_test_dic)
    log_time("Loading feature HYD")
    Read1DFeature(args.tmp_dir + "/HYD.txt", HYD_test_dic)
    log_time("Loading feature PKA")
    Read1DFeature(args.tmp_dir + "/PKA.txt", PKA_test_dic)
    #log_time("Loading feature Pro2Vec_1D")
    #Read1DFeature(args.tmp_dir + "/Pro2Vec_1D.txt", Pro2Vec_1D_test_dic)
    log_time("Loading feature BERT")
    Read1DFeature(args.tmp_dir+"/BERT.txt", BERT_test_dic)
    #log_time("Loading feature BERT full")
    #ReadBERTFeature(args.tmp_dir+"/BERT_n.txt", BERT_test_dic)
    log_time("Loading feature HSP")
    Read1DFeature(args.tmp_dir + "/HSP.txt", HSP_test_dic)
    log_time("Loading feature POSITION")
    Read1DFeature(args.tmp_dir + "/POSITION.txt", POSITION_test_dic)

    log_time("Loading feature PHY_Char")
    Read1DFeature(args.tmp_dir + "/PHY_Char1.txt", PHY_Char_test_dic_1)
    Read1DFeature(args.tmp_dir + "/PHY_Char2.txt", PHY_Char_test_dic_2)
    Read1DFeature(args.tmp_dir + "/PHY_Char3.txt", PHY_Char_test_dic_3)

    log_time("Loading feature PHY_Prop")
    Read1DFeature(args.tmp_dir + "/PHY_Prop1.txt", PHY_Prop_test_dic_1)
    Read1DFeature(args.tmp_dir + "/PHY_Prop2.txt", PHY_Prop_test_dic_2)
    Read1DFeature(args.tmp_dir + "/PHY_Prop3.txt", PHY_Prop_test_dic_3)
    Read1DFeature(args.tmp_dir + "/PHY_Prop4.txt", PHY_Prop_test_dic_4)
    Read1DFeature(args.tmp_dir + "/PHY_Prop5.txt", PHY_Prop_test_dic_5)
    Read1DFeature(args.tmp_dir + "/PHY_Prop6.txt", PHY_Prop_test_dic_6)
    Read1DFeature(args.tmp_dir + "/PHY_Prop7.txt", PHY_Prop_test_dic_7)

    log_time("Loading feature PSSM")
    Read1DFeature(args.tmp_dir + "/PSSM1.txt", PSSM_test_dic_1)
    Read1DFeature(args.tmp_dir + "/PSSM2.txt", PSSM_test_dic_2)
    Read1DFeature(args.tmp_dir + "/PSSM3.txt", PSSM_test_dic_3)
    Read1DFeature(args.tmp_dir + "/PSSM4.txt", PSSM_test_dic_4)
    Read1DFeature(args.tmp_dir + "/PSSM5.txt", PSSM_test_dic_5)
    Read1DFeature(args.tmp_dir + "/PSSM6.txt", PSSM_test_dic_6)
    Read1DFeature(args.tmp_dir + "/PSSM7.txt", PSSM_test_dic_7)
    Read1DFeature(args.tmp_dir + "/PSSM8.txt", PSSM_test_dic_8)
    Read1DFeature(args.tmp_dir + "/PSSM9.txt", PSSM_test_dic_9)
    Read1DFeature(args.tmp_dir + "/PSSM10.txt", PSSM_test_dic_10)
    Read1DFeature(args.tmp_dir + "/PSSM11.txt", PSSM_test_dic_11)
    Read1DFeature(args.tmp_dir + "/PSSM12.txt", PSSM_test_dic_12)
    Read1DFeature(args.tmp_dir + "/PSSM13.txt", PSSM_test_dic_13)
    Read1DFeature(args.tmp_dir + "/PSSM14.txt", PSSM_test_dic_14)
    Read1DFeature(args.tmp_dir + "/PSSM15.txt", PSSM_test_dic_15)
    Read1DFeature(args.tmp_dir + "/PSSM16.txt", PSSM_test_dic_16)
    Read1DFeature(args.tmp_dir + "/PSSM17.txt", PSSM_test_dic_17)
    Read1DFeature(args.tmp_dir + "/PSSM18.txt", PSSM_test_dic_18)
    Read1DFeature(args.tmp_dir + "/PSSM19.txt", PSSM_test_dic_19)
    Read1DFeature(args.tmp_dir + "/PSSM20.txt", PSSM_test_dic_20)

    log_time("Loading features done")


# letter/'1' means interface; '.'/'0' means non-interface
def get_array_of_int_from_a_line(line):
    res = []
    for i in line.rstrip('\n').rstrip(' '):
        if (i == '.' or i == '0'):
            res.append(0)
        else:
            res.append(1)
    return res


def GetProgramArguments():
    logging.info("Parsing program arguments...")
    parser = argparse.ArgumentParser(description='predict.py')
    parser._action_groups.pop()

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input_fn', type=str, required=True,
                          help='str: input protein sequences and label. Format: line1: >p_id. line2: p_seq. line3: label(1:positive, 0: negative)')

    required.add_argument('-d', '--tmp_dir', type=str, required=True,
                          help='str: temporary  directory to store all features. Will be deleted at the end of the program')

    # optional arguments
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-o', '--out_dir', type=str, required=False,
                          help='str: output directory')
    optional.add_argument('-c', '--cpu', type=int, default=1,
                          help='int: use cpu or gpu. 1: cpu; 0: gpu. Default: 1')
    optional.add_argument('-ms', '--model_structure', type=int, default=1,
                          help='int, indicate different models')
    optional.add_argument('-lr', '--learning_rate', type=float, default=0.002,
                          help='float: learning rate')
    optional.add_argument('-do', '--drop_out', type=float, default=0.3,
                          help='float: drop out rate.')
    optional.add_argument('-dn', '--dense', type=int, default=96,
                          help='int, dense unit numbers in the ensemble model')
    optional.add_argument('-bs', '--batch_size', type=int, default=1024, help='int: batch size in learning model')
    optional.add_argument('-unit', '--lstm_unit', type=int, default=32, help='int: number of units in LSTM model')
    optional.add_argument('-ks', '--kernel_size', type=int, default=5, help='int: kernel size in convolution')
    optional.add_argument('-fil', '--filter_size', type=int, default=64, help='int: filter size in convolution')
    optional.add_argument('-ep', '--epochs', type=int, default=7, help='int: number of epochs in training')
    optional.add_argument('-win', '--window_size', type=int, default=31,
                          help='int: the window size of each split of the sequence')
    return parser


def LoadLabelsAndFormatFeatures(args):
#def LoadLabelsAndFormatFeatures(args,batch_index):
    log_time("LoadLabelsAndFormatFeatures")

    ECO_2DList = []
    RAA_2DList = []
    RSA_2DList = []
    #Pro2Vec_1D_2DList = []
    BERT_2DList = []
    Anchor_2DList = []
    HSP_2DList = []
    HYD_2DList = []
    PKA_2DList = []
    POSITION_2DList = []
    PHY_Char_2DList_1 = []
    PHY_Char_2DList_2 = []
    PHY_Char_2DList_3 = []
    PHY_Prop_2DList_1 = []
    PHY_Prop_2DList_2 = []
    PHY_Prop_2DList_3 = []
    PHY_Prop_2DList_4 = []
    PHY_Prop_2DList_5 = []
    PHY_Prop_2DList_6 = []
    PHY_Prop_2DList_7 = []

    PSSM_2DList_1 = []
    PSSM_2DList_2 = []
    PSSM_2DList_3 = []
    PSSM_2DList_4 = []
    PSSM_2DList_5 = []
    PSSM_2DList_6 = []
    PSSM_2DList_7 = []
    PSSM_2DList_8 = []
    PSSM_2DList_9 = []
    PSSM_2DList_10 = []
    PSSM_2DList_11 = []
    PSSM_2DList_12 = []
    PSSM_2DList_13 = []
    PSSM_2DList_14 = []
    PSSM_2DList_15 = []
    PSSM_2DList_16 = []
    PSSM_2DList_17 = []
    PSSM_2DList_18 = []
    PSSM_2DList_19 = []
    PSSM_2DList_20 = []

    label_1DList = []

    fin = open(args.input_fn, "r")
    #index = batch_index
    while True:
    #while (index < batch_index + 500):
        line_PID = fin.readline().rstrip('\n').rstrip(' ')
        line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
        line_label = fin.readline().rstrip('\n').rstrip(' ')
        if not line_label:
            break
        list1D_label_this_line = get_array_of_int_from_a_line(line_label)
        list1D_ECO = ECO_test_dic[line_PID]
        list1D_RAA = RAA_test_dic[line_PID]
        list1D_RSA = RSA_test_dic[line_PID]
        #list1D_Pro2Vec_1D = Pro2Vec_1D_test_dic[line_PID]
        list1D_BERT = BERT_test_dic[line_PID]
        list1D_Anchor = Anchor_test_dic[line_PID]
        list1D_HSP = HSP_test_dic[line_PID]
        list1D_HYD = HYD_test_dic[line_PID]
        list1D_PKA = PKA_test_dic[line_PID]
        list1D_POSITION = POSITION_test_dic[line_PID]
        list1D_PHY_Char_1 = PHY_Char_test_dic_1[line_PID]
        list1D_PHY_Char_2 = PHY_Char_test_dic_2[line_PID]
        list1D_PHY_Char_3 = PHY_Char_test_dic_3[line_PID]
        list1D_PHY_Prop_1 = PHY_Prop_test_dic_1[line_PID]
        list1D_PHY_Prop_2 = PHY_Prop_test_dic_2[line_PID]
        list1D_PHY_Prop_3 = PHY_Prop_test_dic_3[line_PID]
        list1D_PHY_Prop_4 = PHY_Prop_test_dic_4[line_PID]
        list1D_PHY_Prop_5 = PHY_Prop_test_dic_5[line_PID]
        list1D_PHY_Prop_6 = PHY_Prop_test_dic_6[line_PID]
        list1D_PHY_Prop_7 = PHY_Prop_test_dic_7[line_PID]

        list1D_PSSM_1 = PSSM_test_dic_1[line_PID]
        list1D_PSSM_2 = PSSM_test_dic_2[line_PID]
        list1D_PSSM_3 = PSSM_test_dic_3[line_PID]
        list1D_PSSM_4 = PSSM_test_dic_4[line_PID]
        list1D_PSSM_5 = PSSM_test_dic_5[line_PID]
        list1D_PSSM_6 = PSSM_test_dic_6[line_PID]
        list1D_PSSM_7 = PSSM_test_dic_7[line_PID]
        list1D_PSSM_8 = PSSM_test_dic_8[line_PID]
        list1D_PSSM_9 = PSSM_test_dic_9[line_PID]
        list1D_PSSM_10 = PSSM_test_dic_10[line_PID]
        list1D_PSSM_11 = PSSM_test_dic_11[line_PID]
        list1D_PSSM_12 = PSSM_test_dic_12[line_PID]
        list1D_PSSM_13 = PSSM_test_dic_13[line_PID]
        list1D_PSSM_14 = PSSM_test_dic_14[line_PID]
        list1D_PSSM_15 = PSSM_test_dic_15[line_PID]
        list1D_PSSM_16 = PSSM_test_dic_16[line_PID]
        list1D_PSSM_17 = PSSM_test_dic_17[line_PID]
        list1D_PSSM_18 = PSSM_test_dic_18[line_PID]
        list1D_PSSM_19 = PSSM_test_dic_19[line_PID]
        list1D_PSSM_20 = PSSM_test_dic_20[line_PID]

        ECO_2DList.append(list1D_ECO)
        RAA_2DList.append(list1D_RAA)
        RSA_2DList.append(list1D_RSA)
        #Pro2Vec_1D_2DList.append(list1D_Pro2Vec_1D)
        BERT_2DList.append(list1D_BERT)
        Anchor_2DList.append(list1D_Anchor)
        HSP_2DList.append(list1D_HSP)
        HYD_2DList.append(list1D_HYD)
        PKA_2DList.append(list1D_PKA)
        POSITION_2DList.append(list1D_POSITION)
        PHY_Char_2DList_1.append(list1D_PHY_Char_1)
        PHY_Char_2DList_2.append(list1D_PHY_Char_2)
        PHY_Char_2DList_3.append(list1D_PHY_Char_3)
        PHY_Prop_2DList_1.append(list1D_PHY_Prop_1)
        PHY_Prop_2DList_2.append(list1D_PHY_Prop_2)
        PHY_Prop_2DList_3.append(list1D_PHY_Prop_3)
        PHY_Prop_2DList_4.append(list1D_PHY_Prop_4)
        PHY_Prop_2DList_5.append(list1D_PHY_Prop_5)
        PHY_Prop_2DList_6.append(list1D_PHY_Prop_6)
        PHY_Prop_2DList_7.append(list1D_PHY_Prop_7)

        PSSM_2DList_1.append(list1D_PSSM_1)
        PSSM_2DList_2.append(list1D_PSSM_2)
        PSSM_2DList_3.append(list1D_PSSM_3)
        PSSM_2DList_4.append(list1D_PSSM_4)
        PSSM_2DList_5.append(list1D_PSSM_5)
        PSSM_2DList_6.append(list1D_PSSM_6)
        PSSM_2DList_7.append(list1D_PSSM_7)
        PSSM_2DList_8.append(list1D_PSSM_8)
        PSSM_2DList_9.append(list1D_PSSM_9)
        PSSM_2DList_10.append(list1D_PSSM_10)
        PSSM_2DList_11.append(list1D_PSSM_11)
        PSSM_2DList_12.append(list1D_PSSM_12)
        PSSM_2DList_13.append(list1D_PSSM_13)
        PSSM_2DList_14.append(list1D_PSSM_14)
        PSSM_2DList_15.append(list1D_PSSM_15)
        PSSM_2DList_16.append(list1D_PSSM_16)
        PSSM_2DList_17.append(list1D_PSSM_17)
        PSSM_2DList_18.append(list1D_PSSM_18)
        PSSM_2DList_19.append(list1D_PSSM_19)
        PSSM_2DList_20.append(list1D_PSSM_20)

        label_1DList.extend(list1D_label_this_line)

        #index = index + 1
    fin.close()

    ECO_3D_np = Convert2DListTo3DNp(args, ECO_2DList)
    RAA_3D_np = Convert2DListTo3DNp(args, RAA_2DList)
    RSA_3D_np = Convert2DListTo3DNp(args, RSA_2DList)
    #Pro2Vec_1D_3D_np = Convert2DListTo3DNp(args, Pro2Vec_1D_2DList)
    BERT_3D_np = Convert2DListTo3DNp(args, BERT_2DList)
    Anchor_3D_np = Convert2DListTo3DNp(args, Anchor_2DList)
    HSP_3D_np = Convert2DListTo3DNp(args, HSP_2DList)
    HYD_3D_np = Convert2DListTo3DNp(args, HYD_2DList)
    PKA_3D_np = Convert2DListTo3DNp(args, PKA_2DList)
    POSITION_3D_np = Convert2DListTo3DNp(args, POSITION_2DList)
    PHY_Char_3D_np_1 = Convert2DListTo3DNp(args, PHY_Char_2DList_1)
    PHY_Char_3D_np_2 = Convert2DListTo3DNp(args, PHY_Char_2DList_2)
    PHY_Char_3D_np_3 = Convert2DListTo3DNp(args, PHY_Char_2DList_3)
    PHY_Prop_3D_np_1 = Convert2DListTo3DNp(args, PHY_Prop_2DList_1)
    PHY_Prop_3D_np_2 = Convert2DListTo3DNp(args, PHY_Prop_2DList_2)
    PHY_Prop_3D_np_3 = Convert2DListTo3DNp(args, PHY_Prop_2DList_3)
    PHY_Prop_3D_np_4 = Convert2DListTo3DNp(args, PHY_Prop_2DList_4)
    PHY_Prop_3D_np_5 = Convert2DListTo3DNp(args, PHY_Prop_2DList_5)
    PHY_Prop_3D_np_6 = Convert2DListTo3DNp(args, PHY_Prop_2DList_6)
    PHY_Prop_3D_np_7 = Convert2DListTo3DNp(args, PHY_Prop_2DList_7)

    PSSM_3D_np_1 = Convert2DListTo3DNp(args, PSSM_2DList_1)
    PSSM_3D_np_2 = Convert2DListTo3DNp(args, PSSM_2DList_2)
    PSSM_3D_np_3 = Convert2DListTo3DNp(args, PSSM_2DList_3)
    PSSM_3D_np_4 = Convert2DListTo3DNp(args, PSSM_2DList_4)
    PSSM_3D_np_5 = Convert2DListTo3DNp(args, PSSM_2DList_5)
    PSSM_3D_np_6 = Convert2DListTo3DNp(args, PSSM_2DList_6)
    PSSM_3D_np_7 = Convert2DListTo3DNp(args, PSSM_2DList_7)
    PSSM_3D_np_8 = Convert2DListTo3DNp(args, PSSM_2DList_8)
    PSSM_3D_np_9 = Convert2DListTo3DNp(args, PSSM_2DList_9)
    PSSM_3D_np_10 = Convert2DListTo3DNp(args, PSSM_2DList_10)
    PSSM_3D_np_11 = Convert2DListTo3DNp(args, PSSM_2DList_11)
    PSSM_3D_np_12 = Convert2DListTo3DNp(args, PSSM_2DList_12)
    PSSM_3D_np_13 = Convert2DListTo3DNp(args, PSSM_2DList_13)
    PSSM_3D_np_14 = Convert2DListTo3DNp(args, PSSM_2DList_14)
    PSSM_3D_np_15 = Convert2DListTo3DNp(args, PSSM_2DList_15)
    PSSM_3D_np_16 = Convert2DListTo3DNp(args, PSSM_2DList_16)
    PSSM_3D_np_17 = Convert2DListTo3DNp(args, PSSM_2DList_17)
    PSSM_3D_np_18 = Convert2DListTo3DNp(args, PSSM_2DList_18)
    PSSM_3D_np_19 = Convert2DListTo3DNp(args, PSSM_2DList_19)
    PSSM_3D_np_20 = Convert2DListTo3DNp(args, PSSM_2DList_20)

    label_2D_np = np.asarray(label_1DList).reshape(-1, 1)

    assert (
            ECO_3D_np.shape ==
            RAA_3D_np.shape ==
            RSA_3D_np.shape ==
            Anchor_3D_np.shape ==
            HYD_3D_np.shape ==
            PKA_3D_np.shape ==
            PHY_Char_3D_np_1.shape ==
            PHY_Char_3D_np_2.shape ==
            PHY_Char_3D_np_3.shape ==
            PHY_Prop_3D_np_1.shape ==
            PHY_Prop_3D_np_2.shape ==
            PHY_Prop_3D_np_3.shape ==
            PHY_Prop_3D_np_4.shape ==
            PHY_Prop_3D_np_5.shape ==
            PHY_Prop_3D_np_6.shape ==
            PHY_Prop_3D_np_7.shape ==
            #Pro2Vec_1D_3D_np.shape ==
            BERT_3D_np.shape ==
            HSP_3D_np.shape ==
            PSSM_3D_np_20.shape ==
            PSSM_3D_np_1.shape ==
            POSITION_3D_np.shape)

    all_features_3D_np = np.concatenate(
        (ECO_3D_np, RAA_3D_np,
         RSA_3D_np, 
         #Pro2Vec_1D_3D_np,
         BERT_3D_np,
         Anchor_3D_np, 
         HSP_3D_np, 
         HYD_3D_np, 
         PKA_3D_np,
         PHY_Char_3D_np_1, PHY_Char_3D_np_2, PHY_Char_3D_np_3, 
         PHY_Prop_3D_np_1, PHY_Prop_3D_np_2, PHY_Prop_3D_np_3, PHY_Prop_3D_np_4, PHY_Prop_3D_np_5, PHY_Prop_3D_np_6, PHY_Prop_3D_np_7, 
         PSSM_3D_np_1, PSSM_3D_np_2, PSSM_3D_np_3, PSSM_3D_np_4, PSSM_3D_np_5, PSSM_3D_np_6, PSSM_3D_np_7, PSSM_3D_np_8, PSSM_3D_np_9, PSSM_3D_np_10, PSSM_3D_np_11, PSSM_3D_np_12, PSSM_3D_np_13, PSSM_3D_np_14, PSSM_3D_np_15, PSSM_3D_np_16, PSSM_3D_np_17, PSSM_3D_np_18, PSSM_3D_np_19, PSSM_3D_np_20, 
         POSITION_3D_np), axis=2)
		
    #print("Add BERT features")
    #for i in range(0,768):
    	#BERT_tmp = []
    	#for j in range(0,len(BERT_2DList)):
        	#BERT_tmp.append(BERT_2DList[j][i])
    	#BERT_3D_np = Convert2DListTo3DNp(args, BERT_tmp)
    	#all_features_3D_np = np.concatenate((all_features_3D_np, BERT_3D_np), axis=2)

    log_time("Assembling features done.")
    return all_features_3D_np, label_2D_np


def BuildModel(args):
    num_feature = 39
    logging.info("Building model...")
    model = Sequential()
    # the RNN sub-component
    if (int(args.model_structure) == 1):
        input_features = Input(shape=((int)(args.window_size), num_feature))
        out = Bidirectional(
            GRU(name="gru_right", activation="tanh", recurrent_activation="sigmoid", units=args.lstm_unit,
                return_sequences=True, unroll=False, use_bias=True, reset_after=True, recurrent_dropout=0.3),
            name="bidirectional_right")(input_features)
        #out = Bidirectional(CuDNNGRU(name="gru_right", units=args.lstm_unit, return_sequences=True), name="bidirectional_right")(input_features)
        out = Dropout(rate=args.drop_out)(out)
        out = Flatten()(out)
        out = Dense(64, activation='sigmoid', name="dense_RNN_1")(out)
        out = Dropout(rate=args.drop_out)(out)
        out = Dense(1, activation='sigmoid', name="dense_RNN_2")(out)
        model = Model(inputs=input_features, outputs=out)
    # the CNN sub-component
    elif (int(args.model_structure) == 2):
        input_features = Input(shape=((int)(args.window_size), num_feature))
        out = Reshape((args.window_size, num_feature, 1))(input_features)
        out = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, data_format="channels_last",
                     padding="same", activation="relu", name="conv2d_left")(out)
        out = Dropout(rate=args.drop_out)(out)
        out = MaxPool2D(pool_size=3)(out)
        out = Flatten()(out)
        out = Dense(units=args.dense, activation='sigmoid', name="dense_CNN_1")(out)
        out = Dropout(rate=args.drop_out)(out)
        out = Dense(1, activation='sigmoid', name="dense_CNN_2")(out)
        model = Model(inputs=input_features, outputs=out)
    # the ensemble model
    elif (int(args.model_structure) == 3):
        input_features = Input(shape=((int)(args.window_size), num_feature))
        # left
        out1 = Reshape((args.window_size, num_feature, 1))(input_features)
        out1 = Conv2D(filters=args.filter_size, kernel_size=args.kernel_size, name="conv2d_left", trainable=False,
                      padding="same", data_format="channels_last", activation="relu")(out1)
        out1 = MaxPool2D(pool_size=3)(out1)
        out1 = Flatten()(out1)
        out1 = Dropout(rate=args.drop_out)(out1)
        # right
        out2 = Bidirectional(
            GRU(name="gru_right", activation="tanh", recurrent_activation="sigmoid", units=args.lstm_unit,
                return_sequences=True, unroll=False, use_bias=True, reset_after=True, recurrent_dropout=0.3),
            name="bidirectional_right", trainable=False)(input_features)
        #out2 = Bidirectional(CuDNNGRU(name="gru_right", units=args.lstm_unit, return_sequences=True), name="bidirectional_right", trainable=False)(input_features)
        out2 = Dropout(rate=args.drop_out)(out2)
        out2 = Flatten()(out2)
        # merge
        concatenated = Concatenate()([out1, out2])
        out = Dropout(rate=args.drop_out)(concatenated)
        out = Dense(args.dense, activation='sigmoid', name="new_dense1")(out)
        out = Dropout(rate=args.drop_out)(out)
        out = Dense(1, activation='sigmoid', name="new_dense2")(out)
        model = Model(inputs=input_features, outputs=out)
        logging.info("loading weight CNN")
        model.load_weights(args.tmp_dir + "/" + "CNN_DELPHI_v2.h5",by_name=True)
        logging.info("loading weight RNN")
        model.load_weights(args.tmp_dir + "/" + "RNN_DELPHI_v2.h5",by_name=True)
    else:
        logging.error("model_structure error")
        exit(1)
    optimizer_adam = optimizers.Adam(lr=(float)(args.learning_rate), beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['acc'])
    logging.info("Building Model done.")
    model.summary()
    plot_model(model, to_file=args.tmp_dir + "/model_plot.png", show_shapes=True)
    return model


def Train(args, all_features_np3D, all_labels_2D_np):
    if (int(args.model_structure) == 1):
        mc = ModelCheckpoint(args.tmp_dir + "/" + "RNN_DELPHI_v2.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    elif (int(args.model_structure) == 2):
        mc = ModelCheckpoint(args.tmp_dir + "/" + "CNN_DELPHI_v2.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    elif (int(args.model_structure) == 3):
        mc = ModelCheckpoint(args.tmp_dir + "/" + "DELPHI_cpu_HSP_new_BERT_v2.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    else:
        logging.error("model_structure error")
        exit(1)
    # spiting the data into training(9/10) and validation(1/10)
    kfold = KFold(n_splits=10, shuffle=False)
    split_idx = 0
    for train, test in kfold.split(all_features_np3D):
        model = BuildModel(args)
        split_idx += 1
        print("split_idx: ", split_idx)
        # only do validation one time
        if (split_idx > 1):
            break
        # set the class weight of training data because it's imbalanced
        class_weights_protein = class_weight.compute_class_weight('balanced', np.unique(all_labels_2D_np.ravel()),
                                                                  all_labels_2D_np.ravel())
        CV_train_history_protein = model.fit(all_features_np3D[train], all_labels_2D_np[train], callbacks=[mc],
                                             shuffle=True, batch_size=args.batch_size,
                                             class_weight=class_weights_protein, epochs=args.epochs, verbose=2,
                                             validation_data=(all_features_np3D[test], all_labels_2D_np[test]))


def main():
    logging.basicConfig(format='[%(levelname)s] line %(lineno)d: %(message)s', level='INFO')
    log_time("Program started: DELPHI training")
    # Initialize program arguments
    parser = GetProgramArguments()
    args = parser.parse_args()
    print("program arguments are: ", args)
    # Load feature files into feature dictionaries (the global variables below)
    LoadFeatures(args)
    # Format the feature into smaller pieces of size 31, then stack them together. The label handled the same way
    all_features_np3D, all_labels_2D_np = LoadLabelsAndFormatFeatures(args)
    # Train DELPHI
    Train(args, all_features_np3D, all_labels_2D_np)

    #for i in range(0,9845,500):
    	#all_features_np3D, all_labels_2D_np = LoadLabelsAndFormatFeatures(args,i)
    	#Train(args, all_features_np3D, all_labels_2D_np)

    log_time("Program ended")


# global variables.
# below are feature dictionaries. Each dictionary is {key: protein id; value: the numerical values of a given protein}
ECO_test_dic = {}
RAA_test_dic = {}
RSA_test_dic = {}
Pro2Vec_1D_test_dic = {}
BERT_test_dic = {}
Anchor_test_dic = {}
HSP_test_dic = {}
HYD_test_dic = {}
PKA_test_dic = {}
POSITION_test_dic = {}

PHY_Char_test_dic_1 = {}
PHY_Char_test_dic_2 = {}
PHY_Char_test_dic_3 = {}
PHY_Prop_test_dic_1 = {}
PHY_Prop_test_dic_2 = {}
PHY_Prop_test_dic_3 = {}
PHY_Prop_test_dic_4 = {}
PHY_Prop_test_dic_5 = {}
PHY_Prop_test_dic_6 = {}
PHY_Prop_test_dic_7 = {}

PSSM_test_dic_1 = {}
PSSM_test_dic_2 = {}
PSSM_test_dic_3 = {}
PSSM_test_dic_4 = {}
PSSM_test_dic_5 = {}
PSSM_test_dic_6 = {}
PSSM_test_dic_7 = {}
PSSM_test_dic_8 = {}
PSSM_test_dic_9 = {}
PSSM_test_dic_10 = {}
PSSM_test_dic_11 = {}
PSSM_test_dic_12 = {}
PSSM_test_dic_13 = {}
PSSM_test_dic_14 = {}
PSSM_test_dic_15 = {}
PSSM_test_dic_16 = {}
PSSM_test_dic_17 = {}
PSSM_test_dic_18 = {}
PSSM_test_dic_19 = {}
PSSM_test_dic_20 = {}

time_start = time.time()
time_end = time.time()

if __name__ == '__main__':
    main()
