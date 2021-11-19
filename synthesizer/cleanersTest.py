#!/usr/bin/env python3

# Run it like this : cd to this directory then python cleanersTest

from utils.cleaners import french_cleaners, english_cleaners


def test_time() -> None:
    print(english_cleaners("It's 11:00"))
    assert english_cleaners("It's 11:00") == "it's eleven a m"
    assert english_cleaners("It's 9:01") == "it's nine oh one a m"
    assert english_cleaners("It's 16:00") == "it's four p m"
    assert english_cleaners("It's 00:00 am") == "it's twelve a m"


def test_french() -> None:
    print(french_cleaners("J’attendais, suivant la coutume, que la phrase quotidienne fût prononcée."))
    

test_french()
