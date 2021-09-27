from assignment1 import neat_main, ga_main
exp = "neat"


if __name__ == '__main__':
    if exp == "neat":
        neat_main.run()
    elif exp == "ga":
        ga_main.run()
