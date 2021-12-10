# DetailsCNN
Перед запуском необходимо добавить картинки в папки data/train/big, data/train/small, data/test/big, data/test/small.
Для запуска необходимо задать параметры в файле options/opt.yaml. Для только тренировки нейронной сети необходимо задать параметр train : True, test : False, а для только тестирования уже обученной сети train : False, test : True, для тренировки и тестирования параметры train и test оставить неизменными(True).
Далее запустить файл main.py.
Для просмотра графиков точности и ошибки на тесте можно посмотреть файл checkpoints/accuracyavgloss.png.
Модель checkpoints/model.pth даёт точность 100% на тесте.
