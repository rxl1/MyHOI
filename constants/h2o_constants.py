# H2O数据集中的物体名称
h2o_obj_name = [
    '', 'book', 'espresso', 'lotion', 'spray',
    'milk', 'cocoa', 'chips', 'cappuccino', 
]

# 动作列表
action_list = [
    "background",
    "grab book",
    "grab espresso",
    "grab lotion",
    "grab spray",
    "grab milk",
    "grab cocoa",
    "grab chips",
    "grab cappuccino",
    "place book",
    "place espresso",
    "place lotion",
    "place spray",
    "place milk",
    "place cocoa",
    "place chips",
    "place cappuccino",
    "open lotion",
    "open milk",
    "open chips",
    "close lotion",
    "close milk",
    "close chips",
    "pour milk",
    "take out espresso",
    "take out cocoa",
    "take out chips",
    "take out cappuccino",
    "put in espresso",
    "put in cocoa",
    "put in cappuccino",
    "apply lotion",
    "apply spray",
    "read book",
    "read espresso",
    "spray spray",
    "squeeze lotion",
]

# 动词现在分词
present_participle = {
    "grab": "grabbing",  
    "place": "placing", 
    "open": "openning", 
    "close": "closing", 
    "pour": "pouring", 
    "take out": "taking out", 
    "put in": "putting in", 
    "apply": "applying", 
    "read": "reading", 
    "spray": "spraying",
    "squeeze": "squeezing", 
}

# 第三人称动作动词
third_verb = {
    "grab": "grabs", 
    "place": "places", 
    "open": "opens", 
    "close": "closes", 
    "pour": "pours", 
    "take out": "takes take", 
    "put in": "puts in", 
    "apply": "applies", 
    "read": "reads", 
    "spray": "sprays",
    "squeeze": "squeezes", 
}

# 动词过去式
passive_verb = {
    "grab": "is grabbed", 
    "place": "is placed", 
    "open": "is opened", 
    "close": "is closed", 
    "pour": "is poured", 
    "take out": "is taken out", 
    "put in": "is put in", 
    "apply": "is applied", 
    "read": "is read", 
    "spray": "is sprayed",
    "squeeze": "is squeezed", 
}