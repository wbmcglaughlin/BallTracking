from .player import Player


def player_menu():
    name = input("Full Name [first last]: ").lower()
    age = int(input("Age: "))
    sex = input("Sex [male/female]: ").lower()
    height = float(input("Height [m]: "))
    weight = float(input("Weight [kg]: "))
    bowling_hand = False if input("Bowling Hand [L/R]: ") == "L" else True

    Player(name, age, sex, height, weight, bowling_hand)

