# Regulus
CycleGan для состаривания/омоложения людей по фотографии

# Данные

В качестве данных для обучения использовались фотографии знаменитостей из датасета
[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

# Презентация

not ready yet

# Использование
* `python generate_total_csv.py` -- сгенерировать информацию о фотографиях (путь к фотографии, возраст знаменитости)

* `python align.py` -- отцентрировать и кропнуть лица на фотографиях

* `python generate_train_test.py` -- разбить на train/test для CycleGan

* [Обучить CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN)


