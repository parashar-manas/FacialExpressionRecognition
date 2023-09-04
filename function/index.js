// Function to handle file upload and prediction
async function handleFileUpload(event) {
    const file = event.target.files[0];

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send a POST request to the Flask API
        const response = await axios.post('/predict', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });

        // Display the prediction result
        const result = response.data.class;
        document.getElementById('prediction').textContent = `Predicted Emotion: ${result}`;
    } catch (error) {
        console.error(error);
    }
}

// Add an event listener to the file input element
document.getElementById('fileInput').addEventListener('change', handleFileUpload);
