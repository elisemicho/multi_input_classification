# multi_input_classification

## VarDial 2018 Overview Arabic Dialect Identification  
| Rank | Team | Lexical features | Phonetic features | Acoustic features | Model | F1 (macro) | 
| --- | --- | --- | --- | --- | --- | --- |
| 1 | UnibucKernel | x | x | x | KRR on words, KRR on acoustic embs, average | 0.589 |
| 2 | safina | x | | x | GRU+CNN+maxpool+softmax on chars, softmax on acoustic embs, average | 0.576 | 
| 3 | BZU |  |  | x | 2 feed-forward NN+2 multiclass SVMs on acoustic embs, SVM fusion | 0.534 |
| 3 | SYSTRAN | x | x | x | CNN on chars, CNN on phones, concatenate all with acoustic embs, dense layer | 0.529 | 
|  | SYSTRAN | x |  |  | CNN on chars, dense layer | 0.351 | 
|  | SYSTRAN |  | x |  | CNN on phones, dense layer | 0.331 | 
|  | SYSTRAN |  |  | x | acoustic embs, dense layer | 0.521 | 
|  | SYSTRAN | x |  | x | CNN on chars, acoustic embs, dense layer | 0.518 | 
| 3 | Tübingen-Oslo | x | |  | biGRU on chars and words | 0.514 |
| 4 | Arabic_Identification | x | | | Ensemble of SVM classifiers on char and word n-grams | 0.500 |

## Neural network architectures for ADI task in previous VarDial editions
| Year | Team | Lexical features | Acoustic features | Model | F1 (macro) | Rank |
| --- | --- | --- | --- | --- | --- | --- |
| 2017 | deepCybErNetRun | | x | biLSTM on i-vectors | 0.574 | 6/6 |
| 2016 | mitsls | x |  | CNN on chars | 0.483 | 2/18 |
| 2016 | cgli | x |  | CNN on chars | 0.433 | 4/18 |
| 2016 | cgli | x |  | LSTM on chars | 0.423 | |
