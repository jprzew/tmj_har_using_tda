


def get_scorers(labels_dict):

    def labeled_matthews(y_test, y_pred, value):
        return matthews_corrcoef(y_test == value, y_pred == value)

    def labeled_recall(y_test, y_pred, value):
        return recall_score(y_test == value, y_pred == value, zero_division=0)

    # general scorers
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
    matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)

    scorers = {'balanced_accuracy': balanced_accuracy_scorer,
               'matthews': matthews_corrcoef_scorer}

    # matthews correlation scorers
    for label, event in labels_dict.items():
        scoring_function = wrapped_partial(labeled_matthews, value=label)
        scorers[f'matthews_{event}'] = make_scorer(scoring_function)

    # recall scorers
    for label, event in labels_dict.items():
        scoring_function = wrapped_partial(labeled_recall, value=label)
        scorers[f'recall_{event}'] = make_scorer(scoring_function)

    return scorers


def cross_val_evaluate_model(X, y, model, cv, groups=None):

    scorers = get_scorers(cfg.labels_to_events)
    scores = model_selection.cross_validate(model, X, y,
                                            scoring=scorers,
                                            cv=cv, groups=groups, n_jobs=-1)

    return pd.Series(scores)


def robust_cross_val_evaluate_model(X, y, model, cv, groups=None):
    """Evaluates a model and try to trap errors and hide warnings"""

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = cross_val_evaluate_model(X, y, model, cv, groups=groups)
    except:
        scores = None
    return scores



def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    """
    This function will make a pretty plot of a sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    percent:       If True shows percentages. Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for _ in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        perc_matrix = (cf.T / np.sum(cf, axis=1)).T
        group_percentages = ["{0:.0%}".format(value)
                             for value in perc_matrix.flatten()]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels,
                fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def binary_matthews(y_true, y_pred, label):

    return matthews_corrcoef(y_true == label, y_pred == label)