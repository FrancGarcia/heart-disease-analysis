import random

class Patient:
    def __init__(self, age, gender, bmi, blood_pressure, stress_level, 
                 alcohol_consumption, cholesterol_level, diabetes):
        self.age = age
        self.gender = gender
        self.bmi = bmi
        self.blood_pressure = blood_pressure
        self.stress_level = stress_level
        self.alcohol_consumption = alcohol_consumption
        self.cholesterol_level = cholesterol_level
        self.diabetes = diabetes
        
        # Randomly assigned attributes
        self.exercise_habits = random.choice(["Low", "Medium", "High"])
        self.smoking = random.choice(["No", "Yes"])
        self.family_heart_disease = random.choice(["No", "Yes"])
        self.high_blood_pressure = random.choice(["No", "Yes"])
        self.low_hdl_cholesterol = random.choice(["No", "Yes"])
        self.high_ldl_cholesterol = random.choice(["No", "Yes"])
        self.sleep_hours = round(random.uniform(4, 10), 2)
        self.sugar_consumption = random.choice(["Low", "Medium", "High"])
        self.triglyceride_level = random.randint(100, 400)
        self.fasting_blood_sugar = random.randint(80, 141)
        self.crp_level = random.uniform(0, 15)
        self.homocysteine_level = random.uniform(5, 20)
    
    def display_info(self):
        """Displays patient information"""
        print(f"Age: {self.age}")
        print(f"Gender: {self.gender}")
        print(f"BMI: {self.bmi}")
        print(f"Blood Pressure: {self.blood_pressure}")
        print(f"Stress Level: {self.stress_level}")
        print(f"Alcohol Consumption: {self.alcohol_consumption}")
        print(f"Cholesterol Level: {self.cholesterol_level}")
        print(f"Diabetes: {self.diabetes}")
        print(f"Exercise Habits: {self.exercise_habits}")
        print(f"Smoking: {self.smoking}")
        print(f"Family Heart Disease: {self.family_heart_disease}")
        print(f"High Blood Pressure: {self.high_blood_pressure}")
        print(f"Low HDL Cholesterol: {self.low_hdl_cholesterol}")
        print(f"High LDL Cholesterol: {self.high_ldl_cholesterol}")
        print(f"Sleep Hours: {self.sleep_hours:.2f}")
        print(f"Sugar Consumption: {self.sugar_consumption}")
        print(f"Triglyceride Level: {self.triglyceride_level}")
        print(f"Fasting Blood Sugar: {self.fasting_blood_sugar}")
        print(f"CRP Level: {self.crp_level:.2f}")
        print(f"Homocysteine Level: {self.homocysteine_level:.2f}")

    def reduce_risk(self):
        """Suggests steps to reduce heart disease risk based on patient attributes"""
        suggestions = []
        
        # BMI Recommendations
        if self.bmi < 18.5:
            suggestions.append("Your BMI is low. Gain weight through a balanced diet.")
        elif 25 <= self.bmi <= 29.9:
            suggestions.append("Your BMI is overweight. Maintain a healthy weight through regular exercise and diet control.")
        elif self.bmi > 30:
            suggestions.append("Your BMI is in the obesity range. Consider weight loss strategies, including diet and increased physical activity.")
        
        # Blood Pressure Recommendations
        if self.blood_pressure < 120:
            suggestions.append("Your blood pressure is normal.")
        elif 120 <= self.blood_pressure <= 129:
            suggestions.append("Your blood pressure is elevated. Monitor it regularly and maintain a healthy lifestyle.")
        elif 130 <= self.blood_pressure <= 139:
            suggestions.append("Your blood pressure is high. Reduce salt intake and engage in regular exercise.")
        elif self.blood_pressure > 140:
            suggestions.append("Your blood pressure is very high. Consult a doctor for proper management.")
        
        # Stress Level Recommendations
        if self.stress_level == "High":
            suggestions.append("Your stress level is high. Practice relaxation techniques like meditation or deep breathing.")
        
        # Alcohol Consumption Recommendations
        if self.alcohol_consumption in ["Medium", "High"]:
            suggestions.append(f"Your alcohol consumption is {self.alcohol_consumption}. Limit intake to reduce heart disease risk.")
        
        # Cholesterol Level Recommendations
        if self.cholesterol_level < 200:
            suggestions.append("Your cholesterol level is normal.")
        elif 200 <= self.cholesterol_level <= 239:
            suggestions.append("Your cholesterol level is elevated. Reduce saturated fats and cholesterol intake.")
        elif self.cholesterol_level > 240:
            suggestions.append("Your cholesterol level is high. Consult a doctor for proper management.")
        
        # Diabetes Recommendations
        if self.diabetes == "Yes":
            suggestions.append("You have diabetes. Monitor blood sugar levels and maintain a healthy diet.")
        
        # Print Recommendations
        print("Heart Disease Risk Reduction Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
# Example usage
patient1 = Patient(45, "Male", 27.5, 130, "High", "Low", 200, "No")
patient1.reduce_risk()
