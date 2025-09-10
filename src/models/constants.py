TASK_PROMPTS = {
    "llava": {
        "ShortVQA": "\nAnswer the question using a single word or phrase.",
        "ShortCaptioning": "\nCompose a short paragraph of formal analysis for this painting",
        "WikiArtPrompt": "What art style does this painting belongs to?"
    }
    
}

PAINTING_STYLES = [
    'Impressionism',
    'Expressionism',
    'Realism'
]

STYLE_VOCABULARY = [
    "Wet-on-wet",
    "Opaque",
    "Open Composition",
    "Cropped Open",
    "Asymmetrical Open",
    "Snapshot-like",
    "Everyday Life",
    "Landscapes",
    "Modernity",
    "Japonism",
    "Photography",
    "Post-Impressionism",
    "Divisionism",
    "Pointillism",
    "En plein air",
    "Optical Mixing",
    "Fleeting Moments",
    "Broken Color",
    "Short Brushstrokes",
    "Thick Brushstrokes",
    "Loose Brushstrokes ",
    "Visible Brushstrokes ",
    "Dappled Brushstrokes ",
    "Choppy Brushstrokes ",
    "Painterly Brushstrokes",
    "Pure Color",
    "Unmixed Color",
    "Brighter Color",
    "Vibrant Color",
    "Luminous Color",
    "Complementary Color",
    "Contrasting Color",
    "Broken Color",
    "Natural Light",
    "Reflected Natural",
    "Highlights",
    "Shadows",
    "Effects of Light",
    "Changing Atmospheric Conditions"
]

PAINTING_STYLES_DICT = {
    'Impressionism':{
        'predefined_keywords': ["Impasto",
                    "Wet-on-wet",
                    "Opaque",
                    "Open Composition",
                    "Cropped Open",
                    "Asymmetrical Open",
                    "Snapshot-like",
                    "Everyday Life",
                    "Landscapes",
                    "Modernity",
                    "Japonism",
                    "Photography",
                    "Post-Impressionism",
                    "Divisionism",
                    "Pointillism",
                    "En plein air",
                    "Optical Mixing",
                    "Fleeting Moments",
                    "Broken Color",
                    "Short Brushstrokes",
                    "Thick Brushstrokes",
                    "Loose Brushstrokes ",
                    "Visible Brushstrokes ",
                    "Dappled Brushstrokes ",
                    "Choppy Brushstrokes ",
                    "Painterly Brushstrokes",
                    "Pure Color",
                    "Unmixed Color",
                    "Brighter Color",
                    "Vibrant Color",
                    "Luminous Color",
                    "Complementary Color",
                    "Contrasting Color",
                    "Broken Color",
                    "Natural Light",
                    "Reflected Natural",
                    "Highlights",
                    "Shadows",
                    "Effects of Light",
                    "Changing Atmospheric Conditions"]
    }
}