import pandas as pd
import numpy as np
from lifetimes.utils import summary_data_from_transaction_data
from tabulate import tabulate

b_dir = './ecommerce-data/'
transaction_data = pd.read_csv(
    b_dir + 'data.csv', encoding='latin1', dtype={'CustomerID': str, 'InvoiceNo': str})


transaction_data = transaction_data.loc[transaction_data.UnitPrice > 0]
transaction_data = transaction_data.loc[transaction_data.Quantity > 0]

t = transaction_data

t[t.StockCode == '85123A'].groupby('UnitPrice')['Quantity'].sum()
t.groupby('StockCode')['UnitPrice'].unique().apply(len).sort_values()

t[t.StockCode == '79321']
t[t.StockCode == '79321'].groupby('UnitPrice')['Quantity'].sum()
t.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)

train_products = t.groupby("InvoiceNo").apply(
    lambda order: sorted(order['StockCode'].tolist()))
sentences = train_products
longest = np.max(sentences.apply(len))
sentences_ = sentences.values
import gensim
model = gensim.models.Word2Vec(
    sentences_, size=100, window=longest, min_count=2, workers=4)

dic_item_name = t.groupby('StockCode')['Description'].unique().apply(
    lambda x: '\n'.join(x)).to_dict()

p = '79321'


def get_product_recommendation(p):
    suggestions = model.most_similar(p, topn=5)
    print('----', dic_item_name[p])
    list_suggest = []
    for s in suggestions:
        list_suggest.append((s[0], dic_item_name[s[0]], s[1]))
    df = pd.DataFrame(list_suggest)
    df.columns = ['StockCode', 'product_name', 'probability']
    return df


def get_stock_code_by_name(name):
    return [k for k, v in dic_item_name.items() if name.upper() in v]


from tastu_teche.plt_show import df_show, plt_show


def get_product_recommendation_df(name):
    for p in get_stock_code_by_name(name):
        df_show(get_product_recommendation(p), '%s.txt' %
                p, '#product recommendation for %s' % dic_item_name[p])


vocab = list(model.wv.vocab.keys())
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(model.wv.syn0)

import matplotlib.pyplot as plt


def get_batch(vocab, model, n_batches=3):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """From Tensorflow's tutorial."""
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    # plt.show()


embeds = []
labels = []
for item in get_batch(vocab, model, n_batches=3):
    embeds.append(model[item])
    labels.append(dic_item_name[item])
embeds = np.array(embeds)
embeds = pca.fit_transform(embeds)
plot_with_labels(embeds, labels)
plt_show("random_tsne.png")

get_product_recommendation_df('phone')
