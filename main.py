from random import choice

from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from sklearn.neural_network import MLPClassifier

output_notebook()


def get_choice():
    return choice(options)


def find_winner(p1, p2):
    result = 0

    if p1 == p2:
        result = 0

    elif p1 == "piedra" and p2 == "tijeras":
        result = 1
    elif p1 == "piedra" and p2 == "papel":
        result = 2
    elif p1 == "tijeras" and p2 == "piedra":
        result = 2
    elif p1 == "tijeras" and p2 == "papel":
        result = 1
    elif p1 == "papel" and p2 == "piedra":
        result = 1
    elif p1 == "papel" and p2 == "tijeras":
        result = 2

    return result


def str2list(opt):
    if opt == "piedra":
        ret = [1, 0, 0]
    elif opt == "tijeras":
        ret = [0, 1, 0]
    else:
        ret = [0, 0, 1]
    return ret


datax = list(map(str2list, ["piedra","tijeras","papel"]))
datay = list(map(str2list, ["papel","piedra","tijeras"]))
options = ["piedra", "tijeras", "papel"]

clf = MLPClassifier(verbose=False, warm_start=True)
model = clf.fit([datax[0]], [datay[0]])
print(model)


def play(iters=10, debug=False):
    score = {"win": 0, "loose": 0}
    data_x = []
    data_y = []

    for i in range(iters):
        player_1 = get_choice()
        percent = 0.95

        prediction = model.predict_proba([str2list(player_1)])[0]

        if prediction[0] >= percent:
            player_2 = options[0]
        elif prediction[1] >= percent:
            player_2 = options[1]
        elif prediction[2] >= percent:
            player_2 = options[2]
        else:
            player_2 = get_choice()

        if debug:
            try:
                print("Player1: %s Player2 (modelo): %s --> %s" % (player_1, prediction, player_2))
            except:
                print("TypeError: not all arguments converted during string formatting")

        winner = find_winner(player_1, player_2)

        if debug:
            print("Comprobamos: p1 Vs p2: %s" % winner)

        if winner == 2:
            data_x.append(str2list(player_1))
            data_y.append(str2list(player_2))

            score["win"] += 1
        else:
            score["loose"] += 1

    return score, data_x, data_y


i = 0
historic_pct = []
while True:
    i += 1
    score, data_x, data_y = play(iters=1000, debug=False)
    pct = (score["win"] * 100 / (score["win"] + score["loose"]))
    historic_pct.append(pct)
    print("Iter: %s - score: %s %s %%" % (i, score, pct))

    if len(data_x):
        model = model.partial_fit(data_x, data_y)

    if sum(historic_pct[-9:]) == 900:
        break


x = range(len(historic_pct))
y = historic_pct

p = figure(
    title="Porcetaje de aprendizaje en cada iteración",
    x_axis_label="Iter", y_axis_label="%", width=900)

p.line(x, y, legend=None, line_width=1)
show(p)

