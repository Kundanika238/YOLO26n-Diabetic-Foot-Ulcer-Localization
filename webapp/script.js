const imageInput = document.getElementById("imageInput");
const detectBtn = document.getElementById("detectBtn");
const previewBlock = document.getElementById("previewBlock");
const previewImage = document.getElementById("previewImage");
const fileInfo = document.getElementById("fileInfo");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultData = document.getElementById("resultData");

const resultTag = document.getElementById("resultTag");
const errorMessage = document.getElementById("errorMessage");

let selectedFile = null;
let previewURL = null;
let busy = false;

const MAX_SIZE = 10 * 1024 * 1024;

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.hidden = true;
}

function resetResults() {
    resultData.hidden = true;
    loadingState.hidden = true;
    emptyState.hidden = false;
    resultTag.textContent = "Ready";
}

imageInput.addEventListener("change", () => {

    clearError();
    resetResults();

    selectedFile = null;
    detectBtn.disabled = true;
    previewBlock.hidden = true;
    fileInfo.hidden = true;

    if (previewURL) {
        URL.revokeObjectURL(previewURL);
        previewURL = null;
    }

    const file = imageInput.files[0];

    if (!file) return;

    const allowed = [
        "image/jpeg",
        "image/png"
    ];

    if (!allowed.includes(file.type)) {
        showError("Please select a JPG or PNG image.");
        imageInput.value = "";
        return;
    }

    if (file.size > MAX_SIZE) {
        showError("The image must be smaller than 10 MB.");
        imageInput.value = "";
        return;
    }

    selectedFile = file;

    previewURL = URL.createObjectURL(file);
    previewImage.src = previewURL;

    previewBlock.hidden = false;
    fileInfo.hidden = false;
    fileInfo.textContent =
        "Selected: " + file.name +
        " (" + (file.size / 1024 / 1024).toFixed(2) + " MB)";

    detectBtn.disabled = false;
});

function addRegion(index, confidence) {
    const row = document.createElement("div");
    row.className = "region-row";

    const name = document.createElement("span");
    name.textContent = "Potential DFU Region " + index;

    const score = document.createElement("strong");
    score.textContent = confidence.toFixed(2) + "% model score";

    row.appendChild(name);
    row.appendChild(score);

    document.getElementById("regionsList").appendChild(row);
}

function displayResults(data) {

    const count = Number(data.detection_count) || 0;

    const confidence = Number(data.confidence) || 0;

    document.getElementById("resultOriginal").src = previewURL;

    document.getElementById("resultImage").src =
        data.result_image + "?t=" + Date.now();

    document.getElementById("regionCount").textContent = count;

    document.getElementById("highestConfidence").textContent =
        confidence.toFixed(2) + "%";

    const status = document.getElementById("resultStatus");
    const explanation =
        document.getElementById("resultExplanation");

    if (count > 0) {
        status.textContent =
            count + " Potential Region" +
            (count === 1 ? "" : "s") + " Flagged";

        explanation.textContent =
            "The model has identified possible areas " +
            "for further review.";
    } else {
        status.textContent = "No Regions Flagged";

        explanation.textContent =
            "No predictions passed the display threshold. " +
            "This does not establish that the foot is healthy.";
    }

    const regionList =
        document.getElementById("regionsList");

    regionList.replaceChildren();

    if (Array.isArray(data.detections)) {
        data.detections.forEach((item, index) => {
            const score = Number(item.confidence);

            addRegion(
                index + 1,
                Number.isFinite(score) ? score : 0
            );
        });
    }

    document.getElementById("downloadLink").href =
        data.download_url;

    emptyState.hidden = true;
    loadingState.hidden = true;
    resultData.hidden = false;

    resultTag.textContent = "Completed";
}

detectBtn.addEventListener("click", async () => {

    if (!selectedFile || busy) return;

    busy = true;
    detectBtn.disabled = true;
    imageInput.disabled = true;

    clearError();

    emptyState.hidden = true;
    resultData.hidden = true;
    loadingState.hidden = false;

    resultTag.textContent = "Processing";
    detectBtn.textContent = "Analyzing Image...";

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {

        const response = await fetch("/detect", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(
                data.message || "Detection request failed."
            );
        }

        if (!data.result_image || !data.download_url) {
            throw new Error("Incomplete detection result.");
        }

        displayResults(data);

    } catch (error) {

        resetResults();

        showError(
            error.message ||
            "Unable to analyze this image."
        );

    } finally {

        busy = false;
        imageInput.disabled = false;
        detectBtn.disabled = !selectedFile;
        detectBtn.textContent = "Analyze Foot Image";
    }
});
