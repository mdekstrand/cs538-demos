from lenskit.algorithms import basic, bias, user_knn, item_knn, als, svd
import lift, torchmf, torchimf1, torchlmf, torchmfsamp

algorithms = {
    'POP': basic.Popular(),
    'BIAS': bias.Bias(),
    'U-KNN': user_knn.UserUser(30),
    'I-KNN': item_knn.ItemItem(20, 2, save_nbrs=5000),
    'I-KNNi': item_knn.ItemItem(20, 2, feedback='implicit', save_nbrs=5000),
    'E-MF': als.BiasedMF(50, reg=(2, 0.001)),
    'SVD': svd.BiasedSVD(25),
    'I-MF': als.ImplicitMF(50),
    'TorchMF': torchmf.TorchMF(50, reg=0.01),
    'TorchIMF1': torchimf1.TorchImplicitMFUserMSE(50),
    'TorchLMF': torchlmf.TorchLogisticMF(50),
    'SampLMF': torchmfsamp.TorchSampledMF(50),
    'LIFT': lift.Lift(),
}

pred_algos = ['BIAS', 'U-KNN', 'I-KNN', 'E-MF', 'SVD', 'TorchMF']
