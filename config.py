class DataConfig(object):
    """
    Parameters for dataset:

    """
    name = 'vardial2018'
    data_source = 'data/'+name

    def __init__(self):
        print("Data Configurations loaded")


class TrainingConfig(object):
    """
    Parameters for training pipeline:

    """
    epochs = 15
    batch_size = 128
    checkpoint_every = 100

    def __init__(self):
        print("Training Configurations loaded")


####### Default 0.537
# class MultiInputCNNConfig(object):
#     """
#     Parameters for Multi Input CNN model

#     """
#     name = 'multi_input'

#     # Embeddings parameters
#     char_emb_size     = 64
#     phone_CZ_emb_size = 64
#     phone_EN_emb_size = 64
#     phone_HU_emb_size = 64
#     phone_RU_emb_size = 64
#     emb_dropout_p = 0

#     # Convolutional layers parameters
#     filter_sizes = [3,5]
#     num_filters = 16
#     conv_dropout_p = 0

#     # Fully-connected layers parameters
#     fully_connected_layers = [128]
#     dense_dropout_p = 0

#     def __init__(self):
#         print("MultiInputCharCNN Configurations loaded")

####### With Dropout 0.542
# class MultiInputCNNConfig(object):
#     """
#     Parameters for Multi Input CNN model

#     """
#     name = 'multi_input_with_dropout'

#     # Embeddings parameters
#     char_emb_size     = 64
#     phone_CZ_emb_size = 64
#     phone_EN_emb_size = 64
#     phone_HU_emb_size = 64
#     phone_RU_emb_size = 64
#     emb_dropout_p = 0

#     # Convolutional layers parameters
#     filter_sizes = [3,5]
#     num_filters = 16
#     conv_dropout_p = 0.2

#     # Fully-connected layers parameters
#     fully_connected_layers = [128]
#     dense_dropout_p = 0.5

#     def __init__(self):
#         print("MultiInputCharCNN Configurations loaded")

####### Half with dropout 0.547
# class MultiInputCNNConfig(object):
#     """
#     Parameters for Multi Input CNN model

#     """
#     name = 'multi_input_half_with_dropout'

#     # Embeddings parameters
#     char_emb_size     = 32
#     phone_CZ_emb_size = 32
#     phone_EN_emb_size = 32
#     phone_HU_emb_size = 32
#     phone_RU_emb_size = 32
#     emb_dropout_p = 0

#     # Convolutional layers parameters
#     filter_sizes = [3,5]
#     num_filters = 8
#     conv_dropout_p = 0.5

#     # Fully-connected layers parameters
#     fully_connected_layers = [32]
#     dense_dropout_p = 0.5

#     def __init__(self):
#         print("MultiInputCharCNN Configurations loaded")

####### Small with dropout 0.526
# class MultiInputCNNConfig(object):
#     """
#     Parameters for Multi Input CNN model

#     """
#     name = 'multi_input_small_with_dropout'

#     # Embeddings parameters
#     char_emb_size     = 8
#     phone_CZ_emb_size = 8
#     phone_EN_emb_size = 8
#     phone_HU_emb_size = 8
#     phone_RU_emb_size = 8
#     emb_dropout_p = 0.5

#     # Convolutional layers parameters
#     filter_sizes = [3,5]
#     num_filters = 4
#     conv_dropout_p = 0.5

#     # Fully-connected layers parameters
#     fully_connected_layers = [8]
#     dense_dropout_p = 0.5

#     def __init__(self):
#         print("MultiInputCharCNN Configurations loaded")

# ####### Middle with dropout 0.543
# class MultiInputCNNConfig(object):
#     """
#     Parameters for Multi Input CNN model

#     """
#     name = 'multi_input_middle_with_dropout'

#     # Embeddings parameters
#     char_emb_size     = 16
#     phone_CZ_emb_size = 16
#     phone_EN_emb_size = 16
#     phone_HU_emb_size = 16
#     phone_RU_emb_size = 16
#     emb_dropout_p = 0.5

#     # Convolutional layers parameters
#     filter_sizes = [3,5]
#     num_filters = 8
#     conv_dropout_p = 0.5

#     # Fully-connected layers parameters
#     fully_connected_layers = [16]
#     dense_dropout_p = 0.5

#     def __init__(self):
#         print("MultiInputCharCNN Configurations loaded")

# ####### More filters with more dropout 0.524
# class MultiInputCNNConfig(object):
#     """
#     Parameters for Multi Input CNN model

#     """
#     name = 'multi_input_more_filters_with_more_dropout'

#     # Embeddings parameters
#     char_emb_size     = 32
#     phone_CZ_emb_size = 32
#     phone_EN_emb_size = 32
#     phone_HU_emb_size = 32
#     phone_RU_emb_size = 32
#     emb_dropout_p = 0.7

#     # Convolutional layers parameters
#     filter_sizes = [3,5,7]
#     num_filters = 8
#     conv_dropout_p = 0.7

#     # Fully-connected layers parameters
#     fully_connected_layers = [16,16]
#     dense_dropout_p = 0.7

#     def __init__(self):
#         print("MultiInputCharCNN Configurations loaded")

####### Middle with dropout
# class MultiInputCNNConfig(object):
#     """
#     Parameters for Multi Input CNN model

#     """
#     name = 'multi_input_concat_before_conv'

#     # Embeddings parameters
#     char_emb_size     = 32
#     phone_CZ_emb_size = 32
#     phone_EN_emb_size = 32
#     phone_HU_emb_size = 32
#     phone_RU_emb_size = 32
#     emb_dropout_p = 0.5

#     # Convolutional layers parameters
#     filter_sizes = [3,5]
#     num_filters = 8
#     conv_dropout_p = 0.5

#     # Fully-connected layers parameters
#     fully_connected_layers = [32]
#     dense_dropout_p = 0.5

#     def __init__(self):
#         print("MultiInputCharCNN Configurations loaded")

####### Half with dropout
class MultiInputCNNConfig(object):
    """
    Parameters for Multi Input CNN model

    """
    # name = 'multi_input_4phones_only' # 0.364
    # name = 'multi_input_phoneCZ_only' 
    # name = 'multi_input_acoustic_only' # 0.541
    # name = 'multi_input_dense_acoustic' # 0.541
    # name = 'multi_input_char_only' # 0.343
    name = 'multi_input_char_acoustic' # 0.542
    
    # Embeddings parameters
    char_emb_size     = 32
    phone_CZ_emb_size = 32
    phone_EN_emb_size = 32
    phone_HU_emb_size = 32
    phone_RU_emb_size = 32

    # Convolutional layers parameters
    filter_sizes = [3,5]
    num_filters = 8
    conv_dropout_p = 0.5

    # Fully-connected layers parameters
    fully_connected_layers = [32]
    dense_dropout_p = 0.5

    def __init__(self):
        print("MultiInputCharCNN Configurations loaded")