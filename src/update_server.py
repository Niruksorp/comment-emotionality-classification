import requests


def main():
    response = requests.post("http://127.0.0.1:5000/admin/loadModel/")
    if response.status_code != 200:
        for i in range(1, 4):
            rez = requests.post("http://127.0.0.1:5000/admin/loadModel/")
            if rez.status_code == 200:
                break


if __name__ == "__main__":
    main()
