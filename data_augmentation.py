import nlpaug.augmenter.word as naw

original_text = "Rua Amarildo Romari, número 345 jardim gabriel tenório, jundiaí - são paulo, CEP 13245689"

aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = aug.augment(original_text)

print("Texto original:", original_text)
print("Texto aumentado:", augmented_text)