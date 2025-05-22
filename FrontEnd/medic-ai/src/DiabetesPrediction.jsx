import React, { useState, useEffect } from "react";

const DiabetesPrediction = () => {
  const [formData, setFormData] = useState({
    polyuria: "",
    polydipsia: "",
    itching: "",
    gender: "",
    visualBlurring: "",
    suddenWeightLoss: "",
    polyphagia: "",
    genitalThrush: "",
    irritability: "",
    muscleStiffness: "",
    alopecia: "",
    delayedHealing: "",
    partialParesis: "",
    obesity: "",
    weakness: "",
    age: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [explanationImage, setExplanationImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [taskId, setTaskId] = useState(null);

  useEffect(() => {
    const savedTaskId = localStorage.getItem("task_id");
    if (savedTaskId) {
      setTaskId(savedTaskId);
      startPolling(savedTaskId);
    }
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch("http://141.148.194.52:8080/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          Age: parseInt(formData.age),
          Gender: formData.gender,
          Polyuria: formData.polyuria,
          Polydipsia: formData.polydipsia,
          sudden_weight_loss: formData.suddenWeightLoss,
          weakness: formData.weakness,
          Polyphagia: formData.polyphagia,
          Genital_thrush: formData.genitalThrush,
          visual_blurring: formData.visualBlurring,
          Itching: formData.itching,
          Irritability: formData.irritability,
          delayed_healing: formData.delayedHealing,
          partial_paresis: formData.partialParesis,
          muscle_stiffness: formData.muscleStiffness,
          Alopecia: formData.alopecia,
          Obesity: formData.obesity,
        }),
      });

      const data = await response.json();
      setTaskId(data.task_id);
      localStorage.setItem("task_id", data.task_id);
      setPrediction(data.prediction);

      startPolling(data.task_id);
    } catch (error) {
      console.error("Error:", error);
      setLoading(false);
    }
  };

  const startPolling = (task_id) => {
    if (!task_id) return;

    if (window.pollingInterval) {
      clearInterval(window.pollingInterval);
    }

    window.pollingInterval = setInterval(async () => {
      try {
        const response = await fetch(
          `http://141.148.194.52:8080/result/${task_id}`
        );
        const data = await response.json();

        if (data) {
          setExplanation(data.explanation);
          setExplanationImage(data.explanation_image);
        }
      } catch (error) {
        console.error("Error while polling:", error);
      }
    }, 4 * 60 * 1000);
  };

  const featureDescriptions = {
    polyuria:
      "Frequent urination, common in diabetes due to high blood sugar levels.",
    polydipsia:
      "Excessive thirst caused by dehydration due to frequent urination.",
    itching: "Skin itching that can occur when blood sugar levels are high.",
    gender: "Gender can affect the likelihood of developing diabetes.",
    visualBlurring:
      "Blurry vision can occur when high blood sugar alters the lens of the eye.",
    suddenWeightLoss:
      "Unexplained weight loss due to the body's inability to use glucose properly.",
    polyphagia:
      "Excessive hunger because the body isn't able to use glucose effectively.",
    genitalThrush:
      "Fungal infection more common in people with high blood sugar.",
    irritability: "Mood swings caused by fluctuations in blood sugar levels.",
    muscleStiffness:
      "Muscle stiffness due to nerve damage or poor circulation.",
    alopecia:
      "Hair loss often linked to high blood sugar and poor circulation.",
    delayedHealing:
      "Slower healing of wounds due to nerve damage and poor circulation.",
    partialParesis:
      "Weakness or partial paralysis in limbs caused by nerve damage.",
    obesity:
      "Excess body fat, especially abdominal fat, is a major risk factor for diabetes.",
    weakness:
      "Fatigue and muscle weakness from the body’s inability to use glucose.",
    age: "Older individuals are at higher risk of developing diabetes, especially with other risk factors.",
  };

  return (
    <div className="d-flex justify-content-center align-items-start min-vh-100 bg-light p-3">
      <div className="card shadow p-4 w-50" style={{ maxWidth: "500px" }}>
        <div className="card-body">
          <h2 className="text-center mb-4">Diabetes Prediction</h2>
          <form onSubmit={handleSubmit}>
            {Object.keys(formData).map((key) => (
              <div className="mb-3" key={key}>
                <label className="form-label">
                  {key.replace(/([A-Z])/g, " $1").trim()}
                </label>
                {key === "age" ? (
                  <input
                    type="number"
                    name={key}
                    value={formData[key]}
                    onChange={handleChange}
                    className="form-control"
                    placeholder="Enter age"
                  />
                ) : key === "gender" ? (
                  <select
                    name={key}
                    value={formData[key]}
                    onChange={handleChange}
                    className="form-select"
                  >
                    <option value="">Select Gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                ) : (
                  <select
                    name={key}
                    value={formData[key]}
                    onChange={handleChange}
                    className="form-select"
                  >
                    <option value="">Select</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                )}
                <small className="form-text text-muted">
                  {featureDescriptions[key]}
                </small>
              </div>
            ))}
            <button
              type="submit"
              className="btn btn-primary w-100 mt-3"
              disabled={loading}
            >
              {loading ? "Loading..." : "Predict"}
            </button>
          </form>
        </div>
      </div>

      <div className="card shadow p-4 ms-4 w-50">
        <div className="card-body">
          <h2 className="text-center mb-4">Prediction Results</h2>
          {prediction !== null && (
            <div>
              <div className="mt-4 text-center fw-bold">
                Prediction: {console.log(prediction)}
                {prediction === "Diabetes" ? "Diabetic" : "Not Diabetic"}
              </div>
              <div className="mt-4">
                <h5>Explanation:</h5>
                <p>{explanation}</p>
                <h5>Explanation Graph:</h5>
                {explanationImage && (
                  <img
                    src={`data:image/png;base64,${explanationImage}`}
                    alt="Explanation Graph"
                    className="img-fluid"
                  />
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DiabetesPrediction;
