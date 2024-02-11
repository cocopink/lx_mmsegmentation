document.getElementById("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const imageInput = document.getElementById("image");
    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    // 修改这里的URL，将其指向你的服务器上的API
    const apiUrl = "http://<192.168.1.222>:<8060>/predict";
    const response = await fetch(apiUrl, {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    document.getElementById("result").textContent = JSON.stringify(result);
});
