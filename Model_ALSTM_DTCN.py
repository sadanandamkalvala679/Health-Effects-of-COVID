from Evaluate_Error import evaluat_error
from Model_AttentionLSTM import Model_AttentionLSTM
from Model_DTCN import Model_DTCN


def Model_ALSTM_DTCN(trainData, trainTarget, testData, testTarget, sol=None):
    Eval1, pred1 = Model_AttentionLSTM(trainData, trainTarget, testData, testTarget)
    Eval2, pred2 = Model_DTCN(trainData, trainTarget, testData, testTarget)
    pred = (pred1 + pred2) / 2
    Eval = evaluat_error(pred, testTarget)
    return Eval
