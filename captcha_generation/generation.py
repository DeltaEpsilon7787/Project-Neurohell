import claptcha
import os

from random import choice, randint
from shutil import rmtree
from string import ascii_letters, digits

from PIL import Image

full_character_set = ascii_letters + digits
__all__ = ["generate_training_data"]

def generate_training_data(amount, font_directory, training_data_size = (32,32)):
    rmtree('./training_data', ignore_errors=True)
    for character in full_character_set:
        os.makedirs('./training_data/'+str(ord(character)))

    fonts = os.listdir(font_directory)

    for character in full_character_set:
        for i in range(amount):
            another_captcha = None
            while True:
                try:
                    random_font = choice(fonts)
                    another_captcha = claptcha.Claptcha(character,
                                                        os.path.join(font_directory,
                                                                     choice(fonts)
                                                                     )
                                                        )
                except:
                    fonts.remove(random_font)
                    continue
                break
            character_image = another_captcha.image[1].resize(training_data_size)
            character_image.save(os.path.join(
                './training_data',
                str(ord(another_captcha.image[0])),
                str(ord(another_captcha.image[0]))+
                str(randint(0,2**32))+'.bmp'))
