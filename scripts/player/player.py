import json
import os


class Player:
    def __init__(self, name: str, age: int, sex: str, height: float, weight: float, bowling_hand: bool = True):
        if not player_exists(name):
            self.id = new_id()
            self.name = name
            self.age = age
            self.sex = sex
            self.height = height
            self.weight = weight
            self.bowling_hand = bowling_hand
            self.create_profile()

    def create_profile(self):
        profile = {"id": self.id,
                   "name": self.name,
                   "age": self.age,
                   "sex": self.sex,
                   "height": self.height,
                   "weight": self.weight,
                   "bowling_hand": self.bowling_hand,
                   "video_count": 0}

        os.mkdir(f"./profiles/{self.id}")

        with open(f"./profiles/{self.id}/{self.id}.json", "w+") as file:
            json.dump(profile, file)
            os.mkdir(f"./profiles/{self.id}/videos")

        with open("./information/players.json", "r+") as read:
            data = json.load(read)

            data["players"].append({
                "id": self.id,
                "name": self.name
            })

            read.seek(0)
            json.dump(data, read)


def new_id() -> int:
    with open("./information/config.json", "r+") as read:
        data = json.load(read)
        index = int(data["ids"]) + 1
        data["ids"] = index
        read.seek(0)
        json.dump(data, read)
        return index


def player_exists(name: str) -> bool:
    with open("./information/players.json", "r+") as read:
        data = json.load(read)

        if data != {}:
            if any(tag["name"] == name for tag in data["players"]):
                return True

    return False
