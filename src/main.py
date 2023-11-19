from parser import parse
from normalize import normalize
from model import clusterize


def main():
    parse()
    normalize()
    clusterize()


if __name__ == "__main__":
    main()
