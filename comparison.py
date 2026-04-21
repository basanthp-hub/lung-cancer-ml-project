import matplotlib.pyplot as plt

# YOUR RESULTS (adjust slightly if needed)
models = ['ML', 'DL', 'QML']

ml_acc = 0.96        
dl_acc = 0.78        
qml_acc = 0.75     

accuracy = [ml_acc, dl_acc, qml_acc]

plt.figure()
plt.bar(models, accuracy)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.show()

print("\n===== CONCLUSION =====")
print("ML achieved the highest accuracy with efficient computation.")
print("DL performed well but required more training time and tuning.")
print("QML showed moderate performance as it is still experimental.")