## -*- coding: utf-8 -*-
# RTIMBROO UTIL FUNCTIONS

'''

'''
def getFileLogger(logDir, logFileName, name='file_logger', level=20):
    import logging
    #print(level)
    loglevel = None
    if level == 10: # DEBUG
        loglevel = logging.DEBUG;
    elif level == 20: # INFO
        loglevel = logging.INFO;
    elif level == 30: # WARNING
        loglevel = logging.WARNING;
    elif level == 40: # ERROR
        loglevel = logging.ERROR;
    elif level == 50: # CRITICAL
        loglevel = logging.CRITICAL;
    else:
        loglevel = logging.DEBUG;
        
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        
        # logging file handler
        fh = logging.FileHandler(f'{logDir}{logFileName}.log')
        fh.setLevel(logging.DEBUG)
        
        # create console handler with higher level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # create formatter and add it to the handlers
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        
        fh.setFormatter(file_formatter)
        ch.setFormatter(console_formatter)
        
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        # add the handler to the root logger
        #logging.getLogger('').addHandler(ch)
        
        logger.handler_set = True
    return logger


'''
# function used for clocking processing time to build/run models
'''
from contextlib import contextmanager
from timeit import default_timer
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
    

'''
perform label encoding
'''
def labelEncoding(train_df, test_df):
    from sklearn.preprocessing import LabelEncoder
    
    # Label Encoding
    for f in train_df.columns:
        if  train_df[f].dtype=='object': 
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))  
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    
    return(train_df,test_df)


##------------------------------------------------------------------------
# Generate a WordCloud Visualization
# Steps:
#    1: 
# return ...
##------------------------------------------------------------------------
def wordcloud_draw(data, color='black', width=1000, height=750, max_font_size=50, max_words=100):
    import matplotlib.pyplot as plt #2D plotting
    from wordcloud import WordCloud, STOPWORDS
    
    words = ' '.join([str(word) for word in data])
    #cleaned_word = " ".join([word for word in words])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                    background_color=color,
                    width=width,
                    height=height,
                    max_font_size=max_font_size,
                    max_words=max_words,
                     ).generate(words)
    plt.figure(1,figsize=(10.5, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()



'''
# Confusion matrix
'''
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix"', cmap = plt.cm.Blues):
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig(f'{imageDir}explore_sample_images.png', dpi=300)
    plt.show()
    plt = None

'''

'''
def show_time(diff, logger):
    m, s = divmod(diff, 60)
    h, m = divmod(m, 60)
    s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
    logger.info("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))
   
 

'''
# Takes in model scores and plots them on a bar graph

'''
def plot_metric(model_scores, score='Accuracy'):
    import matplotlib.pyplot as plt
    
    # Set figure size
    plt.figure(figsize=(10.5,7))
    plt.bar(model_scores['Model'], height=model_scores[score])
    xlocs, xlabs = plt.xticks()
    xlocs=[i for i in range(0,6)]
    xlabs=[i for i in range(0,6)]
    if(score != 'Prediction Times'):
        for i, v in enumerate(model_scores[score]):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.xlabel('Model')
    plt.ylabel(score)
    plt.xticks(rotation=45)
    plt.show()
    
'''
# RandomForest Modeling
# Takes in training data and a model, and plots a bar graph of the model's feature importances

'''
def feature_importances(df, model, model_name, max_num_features=10):
    import seaborn as sns
    import pandas as pd
    
    feature_importances = pd.DataFrame(columns = ['feature', 'importance'])
    feature_importances['feature'] = df.columns
    feature_importances['importance'] = model.feature_importances_
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    feature_importances = feature_importances[:max_num_features]
    # print(feature_importances)
    plt.figure(figsize=(12, 6));
    sns.barplot(x="importance", y="feature", data=feature_importances);
    plt.title(model_name+' features importance:');

'''

'''
def replace_str_index(text,index=0,replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

'''

'''
def count_features(dic):
    import pandas as pd
    
    bow = []
    # collect kept feature set after cleaning - and count frequencey
    kept_features = {}
    for _id,features in dic.items():
        for word in features:
            bow.append(word)
            if not word in kept_features:
                kept_features[word] = 1
            else:
                word_count = kept_features[word]
                kept_features[word] = word_count+1

    # put the feature word counts into named dictionary and data frame for simpler sorting and observation
    kept_features_named = {'feature':[],'feature_count':[]}
    for feature, count in kept_features.items():
        kept_features_named['feature'].append(feature)
        kept_features_named['feature_count'].append(count)

    # convert dictionary to dataframe for easier sorting
    kept_features_df = pd.DataFrame(kept_features_named)
    kept_features_df_sorted = kept_features_df.sort_values(by=['feature_count','feature'],ascending=False)

    return kept_features_df_sorted,bow

'''

'''
# clean text
def clean_text(logger, text_dic,
                     custom_stop_words=[],
                     remove_pun=True,
                     remove_non_alphabetic=True,
                     remove_stop_words=True,
                     lower_case=False,
                     stemming=False,
                    ):
    import re
    import string
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    # nltk downloads
    nltk.download('punkt')
    
    
    total_tokens_prior = 0
    total_tokens_after = 0
    
    regex_hash=re.compile('^#.+')
    regex_url=re.compile('^http*')

    for _id, tokens in text_dic.items():
        hashes = []
        urls = []
        numbers = []
        non_words = []
        logger.info(f'\ntext: {_id} | feature length prior to text cleaning steps: {len(tokens)}')

        total_tokens_prior = total_tokens_prior+len(text_dic[_id])
        logger.info(f'Total Tokens Prior To Cleaning: {total_tokens_prior}')
        
        try:
            for t in tokens:
                if((re.match(regex_hash,t))):
                    hashes.append(t)

                elif((re.match(regex_url,t))):
                    urls.append(t)

            # remove hash tags
            if len(hashes) > 0:
                # remove these hash tokens from text_tokens
                cleaned_text_tokens = [x for x in tokens if (x not in hashes)]
                text_dic[_id] = cleaned_text_tokens
                tokens = text_dic[_id]
                logger.info(f'text: {_id} | After hash tag removal: {len(tokens)}')

            # remove urls
            if len(urls) > 0:
                cleaned_text_tokens = [x for x in tokens if (x not in urls)]
                text_dic[_id] = cleaned_text_tokens
                tokens = text_dic[_id]
                logger.info(f'text: {_id} | After URL removal: {len(tokens)}')

            # remove punctuation
            if remove_pun:
                table = str.maketrans('','',string.punctuation)
                stripped = [w.translate(table) for w in tokens]
                if len(stripped) > 0:
                    text_dic[_id] = stripped
                    tokens = text_dic[_id]
                    logger.info(f'text: {_id} | After punctuation removal: {len(tokens)}')

            # remove tokens that are not in alphabetic
            if remove_non_alphabetic:
                alpha_words = [word for word in tokens if word.isalpha()]
                if len(alpha_words) > 0:
                    text_dic[_id] = alpha_words
                    tokens = text_dic[_id]
                    logger.info(f'text: {_id} | After non alphabetic removal: {len(tokens)}')
            
            # lower case
            if lower_case:
                lower_words = [word.lower() for word in tokens]
                text_dic[_id] = lower_words
                tokens = text_dic[_id]
                logger.info(f'text: {_id} | After lower case: {len(tokens)}')

            
            # filter out stop words
            if remove_stop_words:
                stop_words = set(stopwords.words('english'))
                new_list = set(list(stop_words) + custom_stop_words)
                not_stop_words = [w for w in tokens if not w in stop_words]
                if len(not_stop_words) > 0:
                    text_dic[_id] = not_stop_words
                    tokens = text_dic[_id]
                    logger.info(f'text: {_id} | After stop word removal: {len(tokens)}')
            
            # consider stemming...???
            if stemming:
                ps = PorterStemmer()
                stem_words = [ps.stem(word) for word in tokens]
                text_dic[_id] = stem_words
                tokens = text_dic[_id]
                logger.info(f'text: {_id} | After stemming: {len(tokens)}')
            
            # count tokens
            total_tokens_after = total_tokens_after+len(text_dic[_id])
            
        except BaseException as be:
            logger.warning(f'**WARNING** Caught BaseException: {be}')
            pass

    logger.info(f'Total Tokens Prior To Cleaning: {total_tokens_prior}')
    logger.info(f'Total Tokens After Cleaning: {total_tokens_after}\n')
    
    
    return text_dic




'''
'''
def inst_vectorizer(ngram_type,vectorizer_type,binary=False,input='filename',max_df=1.0,min_df=1,analyzer='word',max_features=None,stop_words=None,token_pattern=None):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = None
    ngram = (1,1)
    # set ngram type
    if ngram_type == 'unigram':
        ngram = (1,1)
    elif ngram_type == 'bigram':
        ngram = (1,2)
    elif ngram_type == 'trigram':
        ngram = (1,3)
    else:
        ngram = (1,1)
        
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(input=input,binary=binary,ngram_range=ngram,max_df=max_df,min_df=min_df,analyzer=analyzer,max_features=max_features,stop_words=stop_words, token_pattern=token_pattern,decode_error='ignore')
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(input=input,binary=binary,ngram_range=ngram,max_df=max_df,min_df=min_df,analyzer=analyzer,max_features=max_features,stop_words=stop_words, token_pattern=token_pattern,decode_error='ignore')
    else:
        vectorizer = CountVectorizer(input=input,binary=binary,ngram_range=ngram,max_df=max_df,min_df=min_df,analyzer=analyzer,max_features=max_features,stop_words=stop_words, token_pattern=token_pattern,decode_error='ignore')

    return vectorizer


'''

'''
def compress_text_representation(logger, feature_vector):
    word_id_map = {}
    id_word_map = {}

    words = feature_vector.get_feature_names()
    logger.debug(words)

    for i,f in enumerate(words):
        word_id_map[i] = f
        id_word_map[f] = i
    
    return word_id_map, id_word_map



'''
'''
import logging
def train_vector(logger, vector, files, category=None, v_type='count', version=1):
    import pandas as pd
    # Transform the data into a bag of words
    fit = vector.fit(files)
    transform = vector.transform(files)

    features = fit.get_feature_names()

    # print a few of the features
    logger.info(f'Vectorizer[{v_type}], category[{category}], version[{version}] --> transformed shape: {transform.shape}')
    logger.info(f'Vectorizer[{v_type}], category[{category}], version[{version}] --> transformed size: {transform.size}')
    logger.info(f'Vectorizer[{v_type}], category[{category}], version[{version}] --> transformed type: {type(transform)}')
    logger.info(f'Vectorizer[{v_type}], category[{category}], version[{version}] --> vocabulary size: {len(vector.vocabulary_)}')
    
    voc_dict = dict(fit.vocabulary_)
    voc_df = pd.DataFrame.from_dict(voc_dict, orient='index').reset_index()
    voc_df.columns=('feature','feature_index')
    s = voc_df.sort_values(by='feature_index', ascending=False).head(20)
    logger.info(f'Vector Top 20 Head:\n {s}')
    
    
    return fit, transform, features, voc_df


'''
'''
def output_feature_vector(logger, save_as, features, transform_vector, filenames, word_id_map):
    import pandas as pd
    return_code = 0
    
    cols = features
    tdm_vec_df = pd.DataFrame(transform_vector.toarray(),columns=cols)
    non_zero_field_count = 0
    # output feature vector term frequence vector as 'doc feature frequency'
    try:
        with open(f'{save_as}','w+') as f:
            logger.debug(f'output_feature_vector: Opened file for save: {save_as}')
            for i in range(0,tdm_vec_df.shape[0]):
                a = [index for index,value in enumerate(tdm_vec_df.iloc[i]) if value > 0]
                sent = filenames[i]
                logger.debug(f'output_feature_vector: sent: {sent}')
                
                non_zero_field_count = non_zero_field_count+len(a)
                
                for col in a:
                    v = tdm_vec_df.iloc[i,col]
                    logger.debug(f'output_feature_vector: v: {v}')
                    #if col > 1 link words with under score character
                    words = word_id_map[col]
                    
                    words = words.replace(' ','_')
                    
                    sent = sent+' '+words+':'+str(v)
                
                logger.debug(f'output_feature_vector: feature vector output: {sent}')
                f.write(sent+'\n')
                
    except BaseException as be:
        logger.warn(f'**WARNING** Caught Exception writing feature vector to file: {be}')
        return_code = -1
        
        
    return return_code

'''
'''
def create_idf_weights_dict(logger, tfidf_vector, level=logging.INFO):
    import pandas as pd
    
    # get and eval the IDF: The inverse document frequency
    idf = tfidf_vector.idf_
    idf_weights = dict(zip(tfidf_vector.get_feature_names(), idf))
    
    idf_weights_df = pd.DataFrame.from_dict(idf_weights,orient='index').reset_index()
    idf_weights_df.columns=('feature','weight')
    idf_weights_df = idf_weights_df.sort_values(by='weight',ascending=False)
    
    logger.level(f'IDF Top 10 List:\n{idf_weights_df.head(10)}')
    logger.level(f'IDF Lowest 10 List:\n{idf_weights_df.tail(10).sort_values(by="weight",ascending=True)}')
        
    return idf_weights_df

'''
'''
def plot_idf_weights(data, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Set figure size
    plt.figure(figsize=(10.5,7))
    sns.barplot(x='feature', y='weight', data=data)            
    plt.title(title)
    plt.xlabel('Word Features')
    plt.ylabel('IDF Weight')
    plt.show()

'''
'''
def plot_bar(df,x,y,title,ylab,xlab,sort_by,bar_label='Total'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Initialize the matplotlib figure
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(6, 15))
    
    # Plot
    sns.set_color_codes("pastel")
    sns.barplot(x=x, y=y, data=df.sort_values(by=sort_by, ascending=False),
                bar_label="Total", color="b")
    
    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel=ylab,
           xlabel=xlab,
          title=title)
    sns.despine(left=True, bottom=True)
    
    
'''
'''
def plot_train_test_label_split(y_train, y_val, n_classes):
    import matplotlib.pyplot as plt
    import numpy as np
    # number of label classes
    n_classes = n_classes
    
    # look for imbalance in the sample observations for the class
    training_counts = [None] * n_classes
    validation_counts = [None] * n_classes
    
    for i in range(n_classes):
        training_counts[i] = len(y_train[y_train == i])/len(y_train)
        validation_counts[i] = len(y_val[y_val == i])/len(y_val)
    
    # plot histogram of the data
    train_bar = plt.bar(np.arange(n_classes)-0.2, training_counts, align='center', color = 'r', alpha=0.75, width = 0.41, label='Training')
    validate_bar = plt.bar(np.arange(n_classes)+0.2, validation_counts, align='center', color = 'b', alpha=0.75, width = 0.41, label = 'Validating')
    
    plt.xlabel('Labels')
    #plt.xticks((0,1))
    plt.xticks(([i for i in range(0,n_classes)]))
    plt.ylabel('Count (%)')
    plt.title('Label distribution in the training and test set')
    plt.legend(bbox_to_anchor=(1.05, 1), handles=[train_bar, validate_bar], loc=2)
    plt.grid(True)
    plt.show()
    plt = None

'''
'''
def evaluate_predicted_result(logger, feature_vec_path, result_df):
    import pandas as pd
    _df = result_df.copy()
    _df['id'] = result_df.index
    _df_w = _df[['id','True_Label','Predicted_Label']]
    
    test_lines = []
    with open(f'{feature_vec_path}','r') as f:
        lines = f.readlines()
        
        for line in lines:
            tokens = line.split(' ')
            doc = tokens[0]
            doc_toks = doc.split('_')
            doc_index = int(doc_toks[1])
            doc_label = doc_toks[2]
            
            #print(f'doc_index {doc_index}')
            if doc_index in _df_w['id']:
                toks = [t.replace('\n','') for t in tokens]
                test_lines.append({doc_index:toks[1:]})
    
                
    keys = []
    rows = []
    for doc in test_lines:
        for k,v in doc.items():
            keys.append(k)
            rows.append(v)
    
            
    word_frequencys_df = pd.DataFrame()
    word_frequencys_df['id'] = keys
    word_frequencys_df['word_frequency'] = rows
    
    logger.info(word_frequencys_df.head())
    
    result_df_words = pd.merge(_df_w, word_frequencys_df, on='id')
    
    logger.info(result_df_words.head())
    
    return result_df_words
    
'''
'''
def show_most_and_least_informative_features(logger, vectorizer,clf,class_idx=0,n=10):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[class_idx], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[-n:])
    top_feats = []
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        top_feats.append("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
        logger.info("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
    
    return top_feats
'''
'''
def bow_word_frequency(bow):
    import pandas as pd
    
    word_freq = {}
     
    for w in bow:
        if w not in word_freq.keys():
            word_freq[w] = 1
        else:
            word_freq[w] += 1
            
    # sort by frequency then word in desc order
    word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
    word_freq_df.columns=('word','frequency')
    word_freq_df = word_freq_df.sort_values(by=['frequency','word'], ascending=False)
    
    return word_freq_df
    
    
    
'''
Find NaN values in dataframe
'''
def getNaNCount(df):
    
    totNaNCnt = df.isnull().sum().sum()
    nanRowsCnt = len(df[df.isnull().T.any().T])
    
    #print("Total NaN Cnt {0}".format(totNaNCnt))
    #print("Total NaN Rows Cnt {0}".format(nanRowsCnt))
    
    return totNaNCnt, nanRowsCnt
    
'''

'''
def findColumnsNaN(df,rowIndex=True):
    naCols = []
    for col in list(df.columns):
        #print(coachesDf[col].isnull().sum().sum())
        if df[col].isnull().sum().sum() > 0:
            print("Column: {0} has: {1} NaN values".format(col,df[col].isnull().sum().sum()))
            if rowIndex: print("{0}: {1}\n".format(col,getNaNIndexes(df,col)))
            
'''

'''
def getNaNIndexes(df,att):
    import numpy as np
    n = np.where(df[att].isnull()==True)
    return list(n[0])
##---------------------------------------------
    
    
    
    
    
