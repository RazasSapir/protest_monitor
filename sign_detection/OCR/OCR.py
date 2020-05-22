import pytesseract
import cv2
import string
from spellchecker import SpellChecker

WHITE_LIST = string.digits + string.ascii_lowercase + string.ascii_uppercase
LANGUAGES = ['eng', 'heb', 'arb']
pytesseract.pytesseract.tesseract_cmd = r""


def get_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    text = pytesseract.image_to_string(blurred, config="-l " + "+".join(LANGUAGES))
    """ + " -c tessedit_char_whitelist=" + WHITE_LIST)"""
    return text


def eng_spell_checker(sentence):
    spell = SpellChecker()
    formatted_sentence = ""
    split_sentence = sentence.split()
    character_counter = 0
    for word in split_sentence:
        character_counter += len(word)
        formatted_sentence += spell.correction(word).lower()
        if character_counter < len(sentence):  # Adding the delimiter
            formatted_sentence += sentence[character_counter]
        character_counter += 1
    return formatted_sentence


def get_text_from_image(image):
    text = get_ocr(image)
    return text


def main():
    pass


if __name__ == "__main__":
    main()
