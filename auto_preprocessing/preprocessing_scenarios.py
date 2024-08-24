from auto_preprocessing.preprocessing_classes import PreprocessScenario

SCENARIOS = []

class Strategy1:
    def __init__(self, , , , ,)
    все вот это во
    :
    def __proces__
    выполнение всего вот этого

Плюсы: получаем простой json на вход, а не странный .py файл с глобальной переменной, это реально выглядит аккуратнее
+ теперь у нас нет функции eval, которая небезопасна
и jinja тоже теперь нет

Минусы: переписывать код?

template_str = """
        ConvertToGray()
        BinaryThreshold(n_neighbors={{ n_neighbors }}, constant= {{constant}})
        Crop(x_crop=(2, 0.7), y_crop=(0.40, 0.95))
        Resize({{ scale_factor }})
        Dilate({{ dilate_kernel }}, iterations = 1)
        Erode({{ erode_kernel }}, iterations = 1)
"""

param_dict = {
    "scale_factor": ["2"],
    "dilate_kernel": ["np.ones((3,3),int)"],
    "erode_kernel": ["np.ones((3,3),int)"],
    "n_neighbors": ["3", "5", "7", "9", "11"],
    "constant": ["40"],
}
SCENARIOS.append(PreprocessScenario(template_str, param_dict))

template_str2 = """
        ConvertToGray()
        BinaryThreshold(n_neighbors={{ n_neighbors }}, constant= {{constant}})
        Crop(x_crop=(2, 0.7), y_crop=(0.40, 0.95))
"""

param_dict2 = {"n_neighbors": ["3", "5", "7", "9", "11"], "constant": ["10", "20", "30", "40"]}
SCENARIOS.append(PreprocessScenario(template_str2, param_dict2))
