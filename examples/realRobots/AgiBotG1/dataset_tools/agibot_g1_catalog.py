"""Canonical names and source overlays for the AgiBot G1 collection."""

PUBLIC_TASKS = {
    "task_327": "supermarket_shelf_pickup",
    "task_356": "supermarket_bag_packing",
    "task_357": "load_dishwasher",
    "task_358": "toast_bread",
    "task_365": "sort_personal_care_products",
    "task_366": "sort_food",
    "task_367": "remove_toast_from_toaster",
    "task_372": "tote_bag_packing",
    "task_375": "make_tea",
    "task_378": "set_dining_tray",
    "task_388": "supermarket_freezer_pickup",
    "task_390": "supermarket_checkout_scan",
    "task_398": "sort_clothes",
    "task_422": "industrial_logistics_packing",
    "task_424": "clear_tabletop_trash",
    "task_440": "iron_clothes",
}

REAL_TASKS = {
    "AgiBot-g1_pick_mango_and_place_plate": "pick_mango_place_pink_plate",
    "AgiBot-g1_pick_red_pepper_and_place_plate": "pick_red_pepper_place_pink_plate",
    "AgiBot-g1_pick_yellow_pepper_and_place_plate": "pick_yellow_pepper_place_pink_plate",
}

CANONICAL_FIELDS = ["arms", "waist", "head", "grippers"]
STATE_DIMS = {"arms": 14, "waist": 2, "head": 2, "grippers": 2}
ACTION_DIMS = dict(STATE_DIMS)
