from data_loader import load_data

X, y = load_data("data/images", "data/masks")
print("Images shape:", X.shape)
print("Masks shape:", y.shape)
