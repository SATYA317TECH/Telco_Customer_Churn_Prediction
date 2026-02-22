// ========================================
// TELCO CHURN PREDICTOR - MAIN JAVASCRIPT
// ========================================

// ==================== VALIDATION RULES ====================
// Define validation rules for each form field with custom messages
const validationRules = {
    tenure_months: {
        required: true,
        min: 1,
        max: 75,
        requiredMsg: "Please enter tenure months",
        negativeMsg: "Tenure cannot be negative",
        rangeMsg: "Must be between 1-75 months"
    },
    monthly_charges: {
        required: true,
        min: 19,
        max: 119,
        requiredMsg: "Please enter monthly charges",
        negativeMsg: "Monthly charges cannot be negative",
        rangeMsg: "Must be between $19-$119"
    },
    support_ticket_count: {
        required: true,
        min: 0,
        max: 7,
        requiredMsg: "Please enter support tickets count",
        negativeMsg: "Support tickets cannot be negative",
        rangeMsg: "Must be between 0-7 tickets"
    },
    avg_call_minutes: {
        required: true,
        min: 0,
        max: 275,
        requiredMsg: "Please enter average call minutes",
        negativeMsg: "Call minutes cannot be negative",
        rangeMsg: "Must be between 0-275 minutes"
    },
    avg_data_usage_gb: {
        required: true,
        min: 0,
        max: 30,
        requiredMsg: "Please enter average data usage",
        negativeMsg: "Data usage cannot be negative",
        rangeMsg: "Must be between 0-30 GB"
    },
    contract_type: {
        required: true,
        requiredMsg: "Please select a contract type",
        values: ["month-to-month", "one year", "two year"]
    },
    payment_method: {
        required: true,
        requiredMsg: "Please select a payment method",
        values: ["electronic check", "credit card", "bank transfer", "mailed check"]
    }
};

// ==================== GLOBAL VARIABLES ====================
let isFirstPrediction = true;  // Track if this is the first prediction (for button fade effect)

// ==================== VALIDATION FUNCTION ====================
// Validates a single field based on rules
// checkAllRules: false = only check on blur, true = show all errors on submit
function validateField(name, value, element, checkAllRules = false) {
    const msgElement = document.getElementById(`${name}_msg`);
    const rule = validationRules[name];

    if (!rule) return true;

    // Clear previous message and borders
    msgElement.textContent = '';
    msgElement.className = 'validation-message';
    element.classList.remove('error-border', 'valid-border');

    // Check if field is empty
    if (value === null || value === undefined || value === '' || 
        (typeof value === 'string' && value.trim() === '')) {
        if (checkAllRules) {
            msgElement.textContent = rule.requiredMsg;
            msgElement.classList.add('error');
            element.classList.add('error-border');
        }
        return false;
    }

    // Validate number inputs
    if (rule.min !== undefined) {
        const numValue = parseFloat(value);

        // Check if value is a valid number
        if (isNaN(numValue)) {
            if (checkAllRules) {
                msgElement.textContent = "Please enter a valid number";
                msgElement.classList.add('error');
                element.classList.add('error-border');
            }
            return false;
        }

        // Check for negative values
        if (numValue < 0) {
            if (checkAllRules) {
                msgElement.textContent = rule.negativeMsg;
                msgElement.classList.add('error');
                element.classList.add('error-border');
            }
            return false;
        }

        // Check if value is within allowed range
        if (numValue < rule.min || numValue > rule.max) {
            if (checkAllRules) {
                msgElement.textContent = rule.rangeMsg;
                msgElement.classList.add('error');
                element.classList.add('error-border');
            }
            return false;
        }
    }

    // Valid field - add green border
    if (value !== '') {
        element.classList.add('valid-border');
    }
    return true;
}

// ==================== HELPER FUNCTIONS ====================
// Clear all validation messages and reset borders
function clearAllValidations() {
    document.querySelectorAll('.validation-message').forEach(el => {
        el.textContent = '';
        el.className = 'validation-message';
    });
    document.querySelectorAll('input, select').forEach(el => {
        el.classList.remove('error-border', 'valid-border');
    });
}

// Show body loading overlay
function showBodyLoading() {
    document.getElementById('bodyLoading').classList.add('active');
}

// Hide body loading overlay
function hideBodyLoading() {
    document.getElementById('bodyLoading').classList.remove('active');
}

// Trigger validation styling on all fields (used by quick check buttons)
function triggerFieldValidation() {
    document.querySelectorAll('input, select').forEach(element => {
        const event = new Event('blur', { bubbles: true });
        element.dispatchEvent(event);
        
        if (element.tagName === 'SELECT') {
            const changeEvent = new Event('change', { bubbles: true });
            element.dispatchEvent(changeEvent);
        }
    });
}

// ==================== EVENT LISTENERS ====================
// Set up event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dropdowns with gray border
    document.querySelectorAll('select').forEach(select => {
        select.style.borderColor = '#ddd';
        select.style.borderWidth = '1px';
        select.style.borderStyle = 'solid';

        // Update border color when selection changes
        select.addEventListener('change', function() {
            if (this.value) {
                this.style.borderColor = '#2ecc71';
            } else {
                this.style.borderColor = '#ddd';
            }
        });
    });

    // Set up event listeners for all inputs and selects
    document.querySelectorAll('input, select').forEach(element => {
        // Prevent browser's default validation popup
        element.addEventListener('invalid', function(e) {
            e.preventDefault();
            return false;
        });

        // Validate on blur - update border color without showing errors
        element.addEventListener('blur', function() {
            const value = this.type === 'number' ? this.value : this.value;
            const msgElement = document.getElementById(`${this.name}_msg`);
            
            // Clear any existing messages
            if (msgElement) {
                msgElement.textContent = '';
                msgElement.className = 'validation-message';
            }
            
            // Remove error border
            this.classList.remove('error-border');
            
            // Add green border if value is valid
            if (value && value.toString().trim() !== '') {
                const rule = validationRules[this.name];
                if (rule) {
                    if (rule.min !== undefined) {
                        const numValue = parseFloat(value);
                        if (!isNaN(numValue) && numValue >= rule.min && numValue <= rule.max && numValue >= 0) {
                            this.classList.add('valid-border');
                        } else {
                            this.classList.remove('valid-border');
                        }
                    } else {
                        // For dropdowns
                        if (rule.values && rule.values.includes(value)) {
                            this.classList.add('valid-border');
                        } else {
                            this.classList.remove('valid-border');
                        }
                    }
                }
            } else {
                this.classList.remove('valid-border');
            }
        });
        
        // Clear validation styling when user starts typing
        element.addEventListener('input', function() {
            const msgElement = document.getElementById(`${this.name}_msg`);
            if (msgElement) {
                msgElement.textContent = '';
                msgElement.className = 'validation-message';
            }
            this.classList.remove('error-border');
        });
        
        // Handle dropdown changes
        element.addEventListener('change', function() {
            const msgElement = document.getElementById(`${this.name}_msg`);
            if (msgElement) {
                msgElement.textContent = '';
                msgElement.className = 'validation-message';
            }
            
            // Update border for dropdowns
            if (this.value && this.value.toString().trim() !== '') {
                const rule = validationRules[this.name];
                if (rule && rule.values && rule.values.includes(this.value)) {
                    this.classList.add('valid-border');
                } else {
                    this.classList.remove('valid-border');
                }
            } else {
                this.classList.remove('valid-border');
            }
        });
    });
});

// ==================== FORM SUBMIT HANDLER ====================
document.getElementById("churnForm").onsubmit = async function(event) {
    event.preventDefault();
    
    // Clear all previous validations
    clearAllValidations();
    
    // Validate all fields before submitting
    let isValid = true;
    let firstInvalidField = null;
    
    document.querySelectorAll('input, select').forEach(element => {
        const value = element.type === 'number' ? element.value : element.value;
        if (!validateField(element.name, value, element, true)) {
            isValid = false;
            if (!firstInvalidField) {
                firstInvalidField = element;
            }
        }
    });
    
    // If validation fails, scroll to first invalid field
    if (!isValid) {
        if (firstInvalidField) {
            firstInvalidField.scrollIntoView({ behavior: 'smooth', block: 'center' });
            firstInvalidField.focus();
        }
        return false;
    }
    
    // Get DOM elements
    let button = document.getElementById("predictBtn");
    let errorBox = document.getElementById("errorBox");
    let resultHeading = document.getElementById("resultHeading");
    let loading = document.getElementById("loading");
    let resultBox = document.getElementById("resultBox");
    let checkAnotherBtn = document.getElementById("checkAnotherBtn");
    let buttonContainer = document.getElementById("buttonContainer");
    
    // Hide error box and results, show heading and loading spinner
    errorBox.style.display = "none";
    errorBox.textContent = "";
    resultBox.style.display = "none";
    resultHeading.style.display = "block";
    loading.style.display = "block";
    
    // Expand button container to reserve space
    buttonContainer.style.height = "60px";
    buttonContainer.style.marginTop = "10px";
    
    // Set button opacity based on whether this is first prediction
    if (isFirstPrediction) {
        checkAnotherBtn.style.opacity = "0";  // Hidden for first time fade in
    } else {
        checkAnotherBtn.style.opacity = "1";  // Visible for subsequent times
    }
    
    // Disable predict button and show processing text
    button.disabled = true;
    button.textContent = "Processing...";
    
    // Prepare form data for API call
    let formData = new FormData(this);
    const startTime = Date.now();
    const minimumDuration = 1000;  // Minimum spinner display time (ms)
    
    try {
        // Make API call to prediction endpoint
        let response = await fetch("/predict", {
            method: "POST",
            body: formData
        });
        
        let data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Invalid input");
        }
        
        // Ensure spinner shows for minimum duration
        const elapsedTime = Date.now() - startTime;
        if (elapsedTime < minimumDuration) {
            await new Promise(resolve => 
                setTimeout(resolve, minimumDuration - elapsedTime)
            );
        }
        
        // Display results in UI
        document.getElementById("probText").textContent = data.probability + "%";
        
        const riskElement = document.getElementById("riskText");
        riskElement.textContent = data.risk;
        riskElement.className = `risk-badge ${data.risk.toLowerCase()}`;
        
        document.getElementById("suggestionText").textContent = data.suggestion;
        
        // Hide spinner, show results
        loading.style.display = "none";
        resultBox.style.display = "block";
        
        // Handle button fade - only on first prediction
        if (isFirstPrediction) {
            setTimeout(() => {
                checkAnotherBtn.style.opacity = "1";
            }, 50);
            isFirstPrediction = false;  // Mark as not first anymore
        }
        // For subsequent predictions, button is already visible
        
    } catch (err) {
        // Handle errors
        const elapsedTime = Date.now() - startTime;
        if (elapsedTime < minimumDuration) {
            await new Promise(resolve => 
                setTimeout(resolve, minimumDuration - elapsedTime)
            );
        }
        
        errorBox.textContent = "Error: " + err.message;
        errorBox.style.display = "block";
        loading.style.display = "none";
    } finally {
        // Re-enable predict button
        button.disabled = false;
        button.textContent = "Predict Churn";
    }            
};

// ==================== QUICK CHECK FUNCTIONS ====================
// Fill form with low-risk customer example
function fillLowRiskCustomer() {
    const form = document.getElementById("churnForm");
    
    form.querySelector('[name="tenure_months"]').value = 48;
    form.querySelector('[name="contract_type"]').value = "two year";
    form.querySelector('[name="monthly_charges"]').value = 45.00;
    form.querySelector('[name="payment_method"]').value = "credit card";
    form.querySelector('[name="support_ticket_count"]').value = 1;
    form.querySelector('[name="avg_call_minutes"]').value = 275;    
    form.querySelector('[name="avg_data_usage_gb"]').value = 30;
    
    triggerFieldValidation();
}

// Fill form with medium-risk customer example
function fillMediumRiskCustomer() {
    const form = document.getElementById("churnForm");
    
    form.querySelector('[name="tenure_months"]').value = 20;
    form.querySelector('[name="contract_type"]').value = "one year";
    form.querySelector('[name="monthly_charges"]').value = 100.00;
    form.querySelector('[name="payment_method"]').value = "bank transfer";
    form.querySelector('[name="support_ticket_count"]').value = 1;
    form.querySelector('[name="avg_call_minutes"]').value = 250;
    form.querySelector('[name="avg_data_usage_gb"]').value = 25;
    
    triggerFieldValidation();
}

// Fill form with high-risk customer example
function fillHighRiskCustomer() {
    const form = document.getElementById("churnForm");
    
    form.querySelector('[name="tenure_months"]').value = 7;
    form.querySelector('[name="contract_type"]').value = "month-to-month";
    form.querySelector('[name="monthly_charges"]').value = 110.00;
    form.querySelector('[name="payment_method"]').value = "electronic check";
    form.querySelector('[name="support_ticket_count"]').value = 2;
    form.querySelector('[name="avg_call_minutes"]').value = 100;
    form.querySelector('[name="avg_data_usage_gb"]').value = 10;
    
    triggerFieldValidation();
}

// ==================== RESET FORM FUNCTION ====================
// Reset form to initial state with loading animation
async function resetForm() {
    showBodyLoading();

    const minimumDuration = 1000;
    const startTime = Date.now();

    try {
        // Reset form and hide all result sections
        document.getElementById("churnForm").reset();
        document.getElementById("resultBox").style.display = "none";
        document.getElementById("resultHeading").style.display = "none";
        document.getElementById("loading").style.display = "none";
        
        // Collapse button container
        let buttonContainer = document.getElementById("buttonContainer");
        buttonContainer.style.height = "0";
        buttonContainer.style.marginTop = "0";
        
        // Reset flag for next first prediction fade effect
        isFirstPrediction = true;
        
        // Clear all validation messages and borders
        clearAllValidations();

        // Reset all field borders to default
        document.querySelectorAll('input, select').forEach(el => {
            el.classList.remove('error-border', 'valid-border');
            if (el.tagName === "SELECT") {
                el.style.borderColor = '#ddd';
            }
        });

    } finally {
        // Ensure loading shows for minimum duration
        const elapsed = Date.now() - startTime;
        if (elapsed < minimumDuration) {
            await new Promise(resolve =>
                setTimeout(resolve, minimumDuration - elapsed)
            );
        }
        hideBodyLoading();
    }
}

// Expose resetForm to global scope for HTML onclick attribute
window.resetForm = resetForm;