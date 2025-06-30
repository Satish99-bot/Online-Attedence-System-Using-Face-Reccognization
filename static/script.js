const registerButton = document.getElementById("register-button");
const attendanceButton = document.getElementById("attendance-button");
const studentRegistrationContainer = document.getElementById("student-registration-container");
const videoFeedContainer = document.getElementById("video-feed-container");
const uploadContainer = document.getElementById("upload-container");
const video = document.getElementById("cam");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const imageUpload = document.getElementById("image-upload");
const downloadLink = document.getElementById("download-link");
const excelDownload = document.getElementById("excel-download");
const getAttendanceButton = document.getElementById("get-attendance-button");
const studentRegistrationForm = document.getElementById("student-registration-form");
const studentNumberInput = document.getElementById("student-number");
const studentImageInput = document.getElementById("student-image");
let selectedImage = null;
let capturedFrames = [];
let captureInterval = null;

// Toggle views
registerButton.addEventListener("click", () => {
    studentRegistrationContainer.style.display = "flex";
    videoFeedContainer.style.display = "none";
    uploadContainer.style.display = "none";
    clearMessage();
});

attendanceButton.addEventListener("click", () => {
    studentRegistrationContainer.style.display = "none";
    videoFeedContainer.style.display = "flex";
    uploadContainer.style.display = "flex";
    clearMessage();
});

// Image upload handler
imageUpload.addEventListener("change", (event) => {
    clearMessage();
    selectedImage = event.target.files;
    const selectedImagesDiv = document.getElementById("selected-images");
    downloadLink.style.display = "none";
    selectedImagesDiv.innerHTML = "";
    
    if (selectedImage.length > 0) {
        selectedImagesDiv.style.display = "flex";
        Array.from(selectedImage).forEach((file, index) => {
            const container = document.createElement("div");
            container.className = "image-item";
            
            const img = document.createElement("img");
            img.src = URL.createObjectURL(file);
            img.alt = file.name;
            
            const deleteBtn = document.createElement("button");
            deleteBtn.className = "delete-btn";
            deleteBtn.innerHTML = "×";
            deleteBtn.onclick = () => {
                container.remove();
                const updatedFiles = Array.from(selectedImage).filter((_, i) => i !== index);
                const dataTransfer = new DataTransfer();
                updatedFiles.forEach(file => dataTransfer.items.add(file));
                imageUpload.files = dataTransfer.files;
                selectedImage = dataTransfer.files;
                
                if (updatedFiles.length === 0) {
                    selectedImagesDiv.style.display = "none";
                }
            };
            
            container.appendChild(img);
            container.appendChild(deleteBtn);
            selectedImagesDiv.appendChild(container);
        });
    }
});

// Attendance processing
getAttendanceButton.addEventListener("click", async () => {
    if (!selectedImage?.length) {
        showMessage("⚠️ Please upload images first!", false);
        return;
    }

    try {
        showMessage("Processing images...", true);
        const formData = new FormData();
        Array.from(selectedImage).forEach((file, index) => {
            formData.append("images", file);
        });

        const response = await fetch("/process_images", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text().catch(() => "Server error");
            throw new Error(errorText);
        }

        const blob = await response.blob();
        if (blob.size < 1024) throw new Error("Invalid file received");

        const downloadUrl = window.URL.createObjectURL(blob);
        const tempLink = document.createElement('a');
        tempLink.style.display = 'none';
        tempLink.href = downloadUrl;
        tempLink.download = `attendance_${Date.now()}.xlsx`;
        document.body.appendChild(tempLink);
        tempLink.click();

        setTimeout(() => {
            window.URL.revokeObjectURL(downloadUrl);
            tempLink.remove();
            showMessage("✅ Download completed!", true);
        }, 1000);

    } catch (error) {
        const errorMessage = error.message.includes('Failed to fetch') 
            ? "Network error - check connection"
            : error.message;
        showMessage(`❌ Error: ${errorMessage}`, false);
        console.error("Error:", error);
    }
});

// Camera controls
// Camera controls
startButton.addEventListener("click", () => {
    video.src = "/video_feed?" + Date.now();
    showMessage("Camera started", true);
    capturedFrames = [];
    captureInterval = setInterval(() => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        capturedFrames.push(canvas.toDataURL('image/jpeg'));
    }, 2000);
});

stopButton.addEventListener("click", async () => {
    try {
        const response = await fetch("/stop_camera", { method: "POST" });
        if (!response.ok) throw new Error("Failed to stop camera");
        
        video.src = "";
        clearInterval(captureInterval);
        showMessage("Camera stopped", true);

        if (capturedFrames.length > 0) {
            // FIXED: Added missing closing parenthesis
            const frameFiles = await Promise.all(capturedFrames.map(async (dataURL, index) => {
                const blob = await fetch(dataURL).then(r => r.blob());
                return new File([blob], `frame_${index}.jpg`, { type: 'image/jpeg' });
            })); // This line was missing closing )

            const formData = new FormData();
            frameFiles.forEach(file => formData.append('images', file));
            
            const processResponse = await fetch("/process_images", {
                method: "POST",
                body: formData,
            });

            const blob = await processResponse.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = `camera_attendance_${Date.now()}.xlsx`;
            a.click();
        }
        
    } catch (error) {
        showMessage(error.message, false);
        console.error("Error:", error);
    }
});

// Student registration
studentRegistrationForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!studentNumberInput.value || !studentImageInput.files[0]) {
        showMessage("Please fill all fields", false);
        return;
    }

    try {
        const formData = new FormData();
        formData.append("student_number", studentNumberInput.value);
        formData.append("student_image", studentImageInput.files[0]);

        const response = await fetch("/register_student", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error("Registration failed");
        
        studentNumberInput.value = "";
        studentImageInput.value = "";
        showMessage("Student registered successfully!", true);
    } catch (error) {
        showMessage(error.message, false);
        console.error("Error:", error);
    }
});

// Utility functions
function showMessage(message, isSuccess) {
    const container = document.getElementById("message-container");
    container.textContent = message;
    container.style.color = isSuccess ? "green" : "red";
    container.style.display = "block";
    setTimeout(clearMessage, 5000);
}

function clearMessage() {
    const container = document.getElementById("message-container");
    container.textContent = "";
    container.style.display = "none";
}