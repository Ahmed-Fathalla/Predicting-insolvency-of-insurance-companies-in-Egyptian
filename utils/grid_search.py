import operator

from .time_utils import get_TimeStamp_str
from sklearn.model_selection import GridSearchCV

def grid_search(x, y, model, parameters, skf, scoring, verbose=1, wrfile=None, save_res_df=False, n_jobs=-1):
    name_str = get_TimeStamp_str()
    clf = GridSearchCV(model, parameters, cv=skf, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
    clf.fit(x, y)
    res = []
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        res.append((mean, std * 2, params))

    res_str = ''
    # print('-----------', wrfile)
    for i in sorted(res, key=operator.itemgetter(0), reverse=True):
        res_str += '%0.3f (+/-%-.4f) for %r\n' % (i[0], i[1], i[2])

    print(res_str)

    if wrfile is not None:
        with open(wrfile, 'a') as myfile:
            myfile.write(
                '\n\n' + name_str + ' ----------------\n' + res_str + '\n' + get_TimeStamp_str() + ' ----------------\n\n')

    # print(clf.cv_results_)
    if save_res_df:
        df = pd.DataFrame(clf.cv_results_)
        df.to_csv(name_str + '.csv')