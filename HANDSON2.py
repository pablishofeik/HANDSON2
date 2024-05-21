class SimpleLinearRegression:
    def __init__(self):
        self.intercept_ = 0
        self.coef_ = 0
    
    def fit(self, X, y):
        # Calcular la media de X y y
        x_mean = sum(X) / len(X)
        y_mean = sum(y) / len(y)
        
        # Calcular la covarianza y la varianza
        num = 0  
        den = 0  
        for i in range(len(X)):
            num += (X[i] - x_mean) * (y[i] - y_mean)
            den += (X[i] - x_mean) ** 2
        
        # pendiente
        self.coef_ = num / den
        
        # Calcular la intersección
        self.intercept_ = y_mean - self.coef_ * x_mean
    
    def predict(self, X):
        return [self.intercept_ + self.coef_ * x for x in X]
    
    def score(self, X, y):
        # Coeficiente de determinación R^2
        y_pred = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_total = sum((yi - y_mean) ** 2 for yi in y)
        ss_residual = sum((yi - y_pred[i]) ** 2 for i, yi in enumerate(y))
        return 1 - ss_residual / ss_total

    def correlation_coefficient(self, X, y):
        # Coeficiente de correlación
        x_mean = sum(X) / len(X)
        y_mean = sum(y) / len(y)
        num = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(len(X)))
        den = (sum((X[i] - x_mean) ** 2 for i in range(len(X))) * sum((y[i] - y_mean) ** 2 for i in range(len(y)))) ** 0.5
        return num / den
    
# Dataset Benetton
data = [
    {"Year": 1, "Sales": 651, "Advertising": 23},
    {"Year": 2, "Sales": 762, "Advertising": 26},
    {"Year": 3, "Sales": 856, "Advertising": 30},
    {"Year": 4, "Sales": 1063, "Advertising": 34},
    {"Year": 5, "Sales": 1190, "Advertising": 43},
    {"Year": 6, "Sales": 1298, "Advertising": 48},
    {"Year": 7, "Sales": 1421, "Advertising": 52},
    {"Year": 8, "Sales": 1440, "Advertising": 57},
    {"Year": 9, "Sales": 1518, "Advertising": 58},
]

X = [row['Advertising'] for row in data]
y = [row['Sales'] for row in data]


model = SimpleLinearRegression()

model.fit(X, y)

# Imprimir coeficientes
print(f"Intercept: {model.intercept_}")
print(f"Coeficiente: {model.coef_}")

# Imprimir ecuación de regresión
print(f"Ecuación de Regresión: y = {model.intercept_:.2f} + {model.coef_:.2f} * x")

# Predicción 
X_input = 45
y_pred_input = model.predict([X_input])[0]
print(f"Para X = {X_input}, Y predecido = {y_pred_input:.2f}")

# Calcula e imprime los coeficientes de correlación y determinación
correlation_coefficient = model.correlation_coefficient(X, y)
r2_score = model.score(X, y)
print(f"Coeficiente de correlación: {correlation_coefficient:.2f}")
print(f"Coeficiente de determinación (R^2): {r2_score:.2f}")

# Realizar cinco predicciones con datos no existentes
new_X = [60, 65, 70, 75, 80]
new_y_pred = model.predict(new_X)
for i, x in enumerate(new_X):
    print(f"Predicción para X = {x}: Y = {new_y_pred[i]:.2f}")