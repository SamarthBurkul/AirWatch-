// Global toast function
window.showToast = (message, isError = false) => {
    const toast = document.getElementById('toast');
    if (!toast) {
        console.error("Toast element not found!");
        return;
    }
    toast.textContent = message;
    toast.classList.toggle('error', isError);
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
};

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function initAutocomplete(inputId, dropdownId, onSelectCallback) {
    const inputElement = document.getElementById(inputId);
    const dropdownElement = document.getElementById(dropdownId);
    let selectedIndex = -1; // For keyboard navigation

    if (!inputElement || !dropdownElement) {
        console.error(`Autocomplete init failed: Input or Dropdown element not found for ${inputId}/${dropdownId}`);
        return;
    }

    const fetchSuggestions = async (query) => {
        if (query.length < 2) { // Min length check
            dropdownElement.innerHTML = '';
            dropdownElement.style.display = 'none';
            return;
        }

        try {
            const response = await fetch(`/api/autocomplete_city?query=${encodeURIComponent(query)}`);
            if (!response.ok) {
                throw new Error(`Autocomplete API error: ${response.statusText}`);
            }
            const suggestions = await response.json();

            dropdownElement.innerHTML = ''; // Clear previous suggestions
            selectedIndex = -1; // Reset keyboard selection

            if (suggestions.length > 0) {
                suggestions.forEach((suggestion, index) => {
                    const item = document.createElement('div');
                    item.classList.add('autocomplete-item');
                    item.textContent = suggestion;
                    item.dataset.index = index; // Store index for keyboard nav

                    item.addEventListener('click', () => {
                        inputElement.value = suggestion; // Set input value
                        dropdownElement.style.display = 'none'; // Hide dropdown
                        if (onSelectCallback) {
                            onSelectCallback(suggestion); // Call specific page callback
                        }
                    });
                    
                    // Add mouseover for keyboard sync
                    item.addEventListener('mouseover', () => {
                        selectedIndex = index;
                        updateSelectionHighlight();
                    });
                    dropdownElement.appendChild(item);
                });
                dropdownElement.style.display = 'block'; // Show dropdown
            } else {
                dropdownElement.style.display = 'none'; // Hide if no suggestions
            }
        } catch (error) {
            console.error('Failed to fetch city suggestions:', error);
            dropdownElement.innerHTML = '<div class="autocomplete-item text-red-400">Error fetching suggestions</div>';
            dropdownElement.style.display = 'block'; // Show error briefly
            setTimeout(() => { 
                if (dropdownElement.innerHTML.includes('Error')) 
                    dropdownElement.style.display = 'none'; 
            }, 2000);
        }
    };

    // Debounced version of fetchSuggestions
    const debouncedFetch = debounce(fetchSuggestions, 300); // 300ms delay

    inputElement.addEventListener('input', (e) => {
        debouncedFetch(e.target.value);
    });

    // Hide dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!inputElement.contains(e.target) && !dropdownElement.contains(e.target)) {
            dropdownElement.style.display = 'none';
        }
    });

    // Function to update highlight based on selectedIndex
    const updateSelectionHighlight = () => {
        const items = dropdownElement.querySelectorAll('.autocomplete-item');
        items.forEach((item, index) => {
            if (index === selectedIndex) {
                item.classList.add('selected');
                item.scrollIntoView({ block: 'nearest' }); // Scroll into view
            } else {
                item.classList.remove('selected');
            }
        });
    };

    // Keyboard Navigation
    inputElement.addEventListener('keydown', (e) => {
        const items = dropdownElement.querySelectorAll('.autocomplete-item');
        if (items.length === 0 || dropdownElement.style.display === 'none') {
            return;
        }

        if (e.key === 'ArrowDown') {
            e.preventDefault(); // Prevent cursor moving in input
            selectedIndex = (selectedIndex + 1) % items.length;
            updateSelectionHighlight();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault(); // Prevent cursor moving in input
            selectedIndex = (selectedIndex - 1 + items.length) % items.length;
            updateSelectionHighlight();
        } else if (e.key === 'Enter') {
            if (selectedIndex > -1) {
                e.preventDefault(); // IMPORTANT: Prevent default form submission
                items[selectedIndex].click(); // Trigger click on selected item
            }
        } else if (e.key === 'Escape') {
            dropdownElement.style.display = 'none'; // Hide on Escape
        }
    });
}

// ✅ FIXED: Form submission handler - Convert FormData to JSON
document.addEventListener('DOMContentLoaded', () => {
    // Intercept all forms *except* the profile update form
    document.querySelectorAll('form:not([action="/profile"])').forEach(form => {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Ignore predictor forms, they are handled by predictor.js
            if (form.id === 'predict-form' || form.id === 'forecast-form') {
                return;
            }

            const submitBtn = form.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            try {
                // ✅ FIXED: Convert FormData to JSON object
                const formData = new FormData(form);
                const jsonData = {};
                formData.forEach((value, key) => {
                    jsonData[key] = value;
                });

                const response = await fetch(form.action, {
                    method: form.method,
                    headers: {
                        'Content-Type': 'application/json'  // ✅ CRITICAL: Tell server we're sending JSON
                    },
                    body: JSON.stringify(jsonData)  // ✅ FIXED: Send as JSON string
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || `HTTP error! Status: ${response.status}`);
                }

                if (data.success) {
                    showToast(data.message || 'Success!');
                    if (data.redirect) {
                        setTimeout(() => window.location.href = data.redirect, 1000);
                    }
                } else {
                    throw new Error(data.error || 'An unknown error occurred.');
                }
            } catch (error) {
                console.error('Form submission error:', error);
                showToast(error.message, true);
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
            }
        });
    });
});