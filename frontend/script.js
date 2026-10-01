/* ==================================================================
   House Price Predictor — script.js
   ------------------------------------------------------------------
   PHASE 12 scope (this file now):
     - Navbar shadow on scroll
     - Current year in the footer
     - Read the form into a correctly-typed object (numbers vs. text)
     - Validate inputs BEFORE sending (whole numbers, sensible ranges)
     - Send the features to the FastAPI backend (POST /predict)
     - Show the REAL price returned by the trained model
     - Loading state + friendly error messages (network vs. server)
   ================================================================== */

/* The address where your FastAPI backend is running.
   (Phase 11 uses this. Kept at the top so it's easy to change.) */
const API_BASE = "http://127.0.0.1:8000";

/* Grab the elements we'll work with. */
const form          = document.getElementById("predict-form");
const resultEmpty   = document.getElementById("result-empty");
const resultFilled  = document.getElementById("result-filled");
const resultPrice   = document.getElementById("result-price");
const resultRecord  = document.getElementById("result-record");
const formError     = document.getElementById("form-error");
const submitBtn     = document.getElementById("predict-btn");
const nav           = document.getElementById("nav");

const statRows      = document.getElementById("stat-rows");
const statFeatures  = document.getElementById("stat-features");
const statModel     = document.getElementById("stat-model");
const statR2        = document.getElementById("stat-r2");


/* ---- 1. Navbar: add a subtle border/shadow once the page scrolls ---- */
function handleScroll() {
    if (window.scrollY > 8) {
        nav.classList.add("is-scrolled");
    } else {
        nav.classList.remove("is-scrolled");
    }
}
window.addEventListener("scroll", handleScroll);
handleScroll(); // run once on load


/* ---- 2. Footer year ---- */
document.getElementById("year").textContent = new Date().getFullYear();


/* Keep the field notes synchronized with the latest trained model. */
async function loadModelInfo() {
    try {
        const response = await fetch(API_BASE + "/model-info");
        if (!response.ok) return;
        const info = await response.json();
        statRows.textContent = Number(info.rows).toLocaleString();
        statFeatures.textContent = info.features;
        statModel.textContent = info.model;
        statR2.textContent = Number(info.r2).toFixed(2);
    } catch (error) {
        console.error("Could not load model information:", error);
    }
}
loadModelInfo();


/* ---- 3. Read the form into a plain object with correct types ---------
   The backend expects numbers for area/bedrooms/etc. and text for the
   yes/no + furnishing fields. HTML form values are ALWAYS strings, so we
   convert the numeric ones with Number(). This function is reused in
   Phase 11 to build the request body. */
function getFormData() {
    const data = new FormData(form);
    return {
        area:            Number(data.get("area")),
        bedrooms:        Number(data.get("bedrooms")),
        bathrooms:       Number(data.get("bathrooms")),
        stories:         Number(data.get("stories")),
        parking:         Number(data.get("parking")),
        mainroad:        data.get("mainroad"),
        guestroom:       data.get("guestroom"),
        basement:        data.get("basement"),
        hotwaterheating: data.get("hotwaterheating"),
        airconditioning: data.get("airconditioning"),
        prefarea:        data.get("prefarea"),
        furnishingstatus:data.get("furnishingstatus"),
    };
}


/* ---- 4. Show the submitted house details in the result "record" ------ */
function renderRecord(house) {
    const labels = {
        area: "Area", bedrooms: "Bedrooms", bathrooms: "Bathrooms",
        stories: "Stories", parking: "Parking", mainroad: "Main road",
        guestroom: "Guest room", basement: "Basement",
        hotwaterheating: "Hot water", airconditioning: "Air con",
        prefarea: "Pref. area", furnishingstatus: "Furnishing",
    };
    resultRecord.innerHTML = Object.keys(labels).map(function (key) {
        return (
            '<div class="rec"><span class="rec__k">' + labels[key] +
            '</span><span class="rec__v">' + house[key] + "</span></div>"
        );
    }).join("");
}


/* ---- 5. Validate the form BEFORE sending (Phase 12) ------------------
   We check the numeric fields against the SAME limits the backend enforces
   (see backend/app/schemas.py). Catching mistakes here means a friendly,
   instant message instead of a rejected request or a confusing failure.

   The yes/no and furnishing fields are dropdowns, so they can only ever hold
   valid values — no need to check them.

   Returns a human-readable problem string, or null if everything is valid. */
const NUMERIC_LIMITS = {
    area:      { min: 1, max: null, label: "Area" },       // must be > 0
    bedrooms:  { min: 0, max: 20,   label: "Bedrooms" },
    bathrooms: { min: 0, max: 20,   label: "Bathrooms" },
    stories:   { min: 0, max: 20,   label: "Stories" },
    parking:   { min: 0, max: 20,   label: "Parking spaces" },
};

function validateForm() {
    const data = new FormData(form);

    for (const key in NUMERIC_LIMITS) {
        const rule = NUMERIC_LIMITS[key];
        const raw  = (data.get(key) || "").trim();

        if (raw === "") {
            return "Please fill in the " + rule.label + " field.";
        }
        const value = Number(raw);
        if (Number.isNaN(value)) {
            return rule.label + " must be a number.";
        }
        if (!Number.isInteger(value)) {
            return rule.label + " must be a whole number (no decimals).";
        }
        if (value < rule.min) {
            return rule.label + " must be at least " + rule.min + ".";
        }
        if (rule.max !== null && value > rule.max) {
            return rule.label + " must be " + rule.max + " or less.";
        }
    }
    return null; // all fields are valid
}


/* ---- 6. Handle form submit ------------------------------------------
   PHASE 12 version: validate first, then send to the FastAPI backend and
   display the REAL price the model returns.

   Important project rule: we NEVER invent or hard-code a price. The number
   shown here comes straight from the trained model via POST /predict.

   We use async/await so the code reads top-to-bottom even though the network
   call takes a moment. */
form.addEventListener("submit", async function (event) {
    event.preventDefault(); // stop the page from reloading

    // 1) Check the inputs BEFORE contacting the server.
    const problem = validateForm();
    if (problem) {
        formError.hidden = false;
        formError.textContent = problem;
        return; // stop here — don't send bad data
    }

    const house = getFormData();

    // 2) Reveal the filled state, show the details, and start a loading state.
    resultEmpty.hidden = true;
    resultFilled.hidden = false;
    renderRecord(house);

    formError.hidden = true;
    formError.textContent = "";
    submitBtn.classList.add("is-loading");
    submitBtn.disabled = true;
    resultPrice.textContent = "Estimating…";

    // 3) Send the house features to the backend and handle the response.
    try {
        const response = await fetch(API_BASE + "/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(house),
        });

        // If the server replied with an error status (e.g. 422, 500, 503),
        // read its explanation so we can show something useful.
        if (!response.ok) {
            let message = "The server rejected the request (status " +
                          response.status + ").";
            try {
                const errBody = await response.json();
                if (errBody && errBody.detail) {
                    if (Array.isArray(errBody.detail)) {
                        // FastAPI/Pydantic validation errors come as a list.
                        message = "Please check your inputs: " +
                            errBody.detail.map(function (d) { return d.msg; })
                                          .join("; ");
                    } else {
                        // 500 / 503 send a single text message.
                        message = String(errBody.detail);
                    }
                }
            } catch (parseErr) {
                // Response body wasn't JSON — keep the default message.
            }
            throw new Error(message);
        }

        // Success. The backend returns:
        //   { predicted_price: 7211633.5, formatted_price: "Rs 7,211,633" }
        const result = await response.json();

        // Show the formatted price, using the ₹ symbol instead of "Rs".
        resultPrice.textContent = result.formatted_price.replace("Rs", "₹");
    } catch (error) {
        console.error("Prediction request failed:", error);
        resultPrice.textContent = "₹ —";
        formError.hidden = false;

        // A network failure (server not running / unreachable) throws a
        // TypeError. Anything else is an error status we described above.
        if (error instanceof TypeError) {
            formError.textContent =
                "Could not reach the prediction server. Make sure the backend " +
                "is running at " + API_BASE + " and try again.";
        } else {
            formError.textContent = error.message;
        }
    } finally {
        // Always turn the loading state off, whether we succeeded or failed.
        submitBtn.classList.remove("is-loading");
        submitBtn.disabled = false;
    }
});
