from ege_parser.preprocessing_classes import PreprocessScenario

SCENARIOS = []

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
