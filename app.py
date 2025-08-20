from flask import Flask, render_template
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
from flask import request
from PIL import Image


app = Flask(__name__)

# Model ve veri yükle
model = load_model("harf_tanima_modeli.h5")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

@app.route("/")
def index():
    return render_template("index.html")



def hazirla_ve_tahmin_et_cv2(model, img_array):
    import cv2
    import numpy as np

    # Gri tonlamaya çevir
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Adaptive Threshold (arka plan farklılıklarına karşı daha dayanıklı)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Gürültü temizleme (morfolojik işlem)
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # En büyük konturu al
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Karakter bulunamadı", None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    char_crop = thresh[y:y+h, x:x+w]

    # Harfi merkeze al ve 28x28 boyutuna getir
    resized = cv2.resize(char_crop, (22, 22))
    padded = np.pad(resized, ((3, 3), (3, 3)), 'constant', constant_values=0)
    final_img = padded / 255.0
    final_img = final_img.reshape(1, 28, 28, 1)

    prediction = model.predict(final_img)
    predicted_label = np.argmax(prediction)
    return chr(predicted_label + 65), padded




@app.route("/kamera_tahmin", methods=["POST"])
def kamera_tahmin():
    if "image" not in request.files:
        return "Görsel bulunamadı", 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img_array = np.array(img)

    sonuc, gorsel = hazirla_ve_tahmin_et_cv2(model, img_array)

    if gorsel is not None:
        cv2.imwrite("static/img.png", gorsel)

    return render_template("index.html", prediction=sonuc, gercek="?" if sonuc else "Tanımsız")

@app.route("/tahmin")
def tahmin():
    i = np.random.randint(0, len(X_test))
    img = X_test[i].reshape(1, 28, 28, 1)
    true_label = np.argmax(y_test[i])
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)

    # Görseli kaydet (static klasörüne)
    plt.imsave("static/img.png", X_test[i].reshape(28,28), cmap="gray")

    return render_template(
        "index.html",
        prediction=chr(predicted_label + 65),
        gercek=chr(true_label + 65)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
