document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('prediction-result');
    const errorDiv = document.getElementById('error-message');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Hide previous results and errors
        resultDiv.textContent = '';
        errorDiv.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        loadingDiv.classList.remove('hidden');
        
        // Gather form data
        const formData = {
            location: document.getElementById('location').value.trim(),
            area: parseFloat(document.getElementById('area').value),
            bedrooms: parseInt(document.getElementById('bedrooms').value),
            bathrooms: parseInt(document.getElementById('bathrooms').value),
            balcony: parseInt(document.getElementById('balcony').value),
            property_type: document.getElementById('property_type').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            
            loadingDiv.classList.add('hidden');

            if (response.ok) {
                resultDiv.textContent = data.formatted_price;
            } else {
                showError(data.error || 'An error occurred during prediction.');
            }
        } catch (error) {
            loadingDiv.classList.add('hidden');
            showError('Failed to connect to the server. Please try again later.');
            console.error('Prediction Error:', error);
        }
    });

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
    }
});
