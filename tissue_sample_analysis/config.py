BASE_DIR = "../data"
IMAGES_DIR = "../data/images"
EXTRACTED_PATCHES_DIR = "../data/patches"
PROTEIN_EXPRESSIONS_CSV = "../data/protein_expressions_spotlevel.csv"

# Sample A: N540
# Sample B: N935
# Sample C: N403
# Sample D: N1027

SAMPLE_CONFIG = {
    'A': {
        #"file": "n540_19a2_EMSC.mat",
        "file": "n540_19a2_derivative_1.mat",
        "hne_coordinates": [[788, 971], [177, 1115], [449, 895], [1118, 995], [1332, 1319], [912, 1231], [652, 289], [993, 361],[1216, 794], [1153, 567], [528, 1252], [161, 497],
                            [1021, 1039], [724, 219], [811, 208], [707, 704], [696, 548], [404, 775], [391, 995], [560, 846], [572, 748], [360, 814], [509, 1143], [375, 1094], [932, 60]],
        "spectral_coordinates": [[743, 933], [63, 951], [384, 781], [1088, 1027], [1249, 1378], [808, 1235], [768, 178], [1077, 296], [1170, 801], [1181, 574], [417, 1167], [185, 272],
                                 [979, 1034], [833, 128], [895, 118], [709, 621], [712, 437], [395, 643], [315, 864], [539, 742], [550, 651], [322, 686], [422, 1072], [280, 967], [1071, 21]],
        "hne_full_width": 17966, "hne_full_height": 22915,
        "ir_width": 1382, "ir_height": 1762,
        "win_hne": 256, "win_ir": 20,
        "images_dir": IMAGES_DIR
    },
    'B': {
        #"file": "n935_19b1_EMSC.mat",
        "file": "n935_19b1_derivative_1.mat",
        "hne_coordinates": [[663, 857], [614, 900], [647, 928], [544, 655], [342, 767], [820, 747], [714, 635], [379, 640], [893, 854], [464, 1060], [734, 941], [448, 765], [632, 1008], [352, 980], 
                            [400, 737], [838, 860]],  
        "spectral_coordinates": [[438, 450], [408, 497], [435, 555], [323, 155], [73, 274], [669, 367], [560, 190], [136, 83], [729, 473], [198, 694], [528, 558], [193, 290], [383, 634], [47, 548], 
                                 [174, 271], [684, 487]],
        "e_full_width": 15911, "e_full_height": 16689,  
        "ir_width": 1060, "ir_height": 1112, 
        "win_hne": 256, "win_ir": 20,
        "images_dir": IMAGES_DIR
    },
    'C': {
        #"file": "n403_18e2_EMSC.mat",
        "file": "n403_18e2_derivative_1.mat",
        "hne_coordinates": [[304, 995], [428, 1186], [1060, 1086], [204, 727], [1085, 205], [798, 152], [1096, 347], [575, 112], [222, 1061], [1122, 97], [1216, 65], [512, 1147], [1176, 438],
                            [1208, 443], [209, 498], [173, 537], [149, 561], [1107, 586], [518, 78], [449, 197], [453, 398], [463, 353], [537, 1160], [503, 1205], [366, 1163], [584, 1149], 
                            [1000, 1126], [875, 145], [926, 273], [886, 267]],
        "spectral_coordinates": [[121, 931], [225, 1161], [909, 1158], [66, 643], [1083, 233], [822, 147], [1069, 393], [536, 73], [24, 978], [1139, 130], [1242, 100], [304, 1127], [1138, 498], 
                                 [1176, 528], [119, 419], [67, 454], [41, 473], [10, 479], [424, 21], [404, 141], [388, 353], [385, 306], [334, 1148], [272, 1189], [143, 1112], [385, 1146],
                                 [828, 1187], [877, 146], [917, 304], [878, 290]],
        "hne_full_width": 18467, "hne_full_height": 16665,
        "ir_width": 1420, "ir_height": 1281,
        "win_hne": 256, "win_ir": 20,
        "images_dir": IMAGES_DIR
    },
    'D': {
        #"file": "n1027_19a1_EMSC.mat",
        "file": "n1027_19a1_derivative_1.mat",
        "hne_coordinates": [[163, 500], [237, 248], [432, 197], [622, 373], [600, 406], [834, 406], [504, 803], [352, 739], [226, 790], [234, 551], [428, 496], [664, 809], [598, 601]],
        "spectral_coordinates": [[63, 404], [197, 83], [485, 46], [698, 295], [661, 334], [972, 380], [497, 870], [286, 745], [109, 786], [155, 480], [418, 430], [712, 891], [631, 597]],
        "e_full_width": 15905, "e_full_height": 15768,
        "ir_width": 1060, "ir_height": 1051,
        "win_hne": 256, "win_ir": 20,
        "images_dir": IMAGES_DIR
    }
}


# Samples A and C registered to H&E image; Samples B and D registered to E-image
def get_sample_info(sample_id):
    info = SAMPLE_CONFIG[sample_id]

    # Calculate scale factors
    if sample_id in ['A', 'C']:
        info["scale_factor_x"] = info["hne_full_width"] / info["ir_width"]
        info["scale_factor_y"] = info["hne_full_height"] / info["ir_height"]
    else:  # Sample B and D
        info["scale_factor_x"] = info["e_full_width"] / info["ir_width"]
        info["scale_factor_y"] = info["e_full_height"] / info["ir_height"]

    # Calculate window sizes
    info["hw_hne"] = info["win_hne"] // 2
    info["hw_ir"] = (info["win_ir"] + 1) // 2

    return info
