const imgSelector = document.getElementById("currentImageFileSelector");
const containerInputScreen2 = document.getElementById("fileInputScreen2");
const containerInputScreen1 = document.getElementById("fileInputScreen1");

const containerSwithShowRawImage =
  document.getElementById("switchShowRawImage");
const containerClearDbAndStorage = document.getElementById(
  "BtnCleanerDatabaseAndImageStorage"
);
const containerLoading = document.getElementById("loading");
const containerScreen2 = document.getElementById("screen2");
const containerScreen1 = document.getElementById("screen1");

let urlImagesStoreDir = "http://localhost:8000/results";

console.log(`imgSelector before ${imgSelector.value}`);

function genHtmlRawOption(fileName) {
  const option = document.createElement("option");
  const filename_without_hash = fileName.slice(33);
  option.setAttribute("value", fileName);
  option.text = filename_without_hash;
  option.className = "image-selector-option";
  return option;
}

async function refreshFileSelector() {
  containerLoading.classList.remove("hidden");
  containerScreen2.classList.add("hidden");
  const response = await fetch("/api/get_all_uploaded_files", {
    method: "GET",

    headers: { Accept: "application/json" },
  });
  if (response.ok) {
    const countOfUploadedFiles = await response.json();
    if (countOfUploadedFiles.message !== "base_is_empty") {
      imgSelector.innerHTML = "";
      countOfUploadedFiles.forEach((fileName) =>
        imgSelector.append(genHtmlRawOption(fileName))
      );
      containerScreen2.classList.remove("hidden");
      containerLoading.classList.add("hidden");
      let length = countOfUploadedFiles.length;
      selector = document.getElementById("currentImageFileSelector");
      selector.selectedIndex = length - 1;
      selector.dispatchEvent(new Event('change'));

    } else {
      containerScreen1.classList.remove("hidden");
      containerScreen2.classList.add("hidden");
      containerLoading.classList.add("hidden");
    }
  } else {
    containerScreen1.classList.remove("hidden");
    containerScreen2.classList.add("hidden");
    containerLoading.classList.add("hidden");
  }
}

function addMoveBtns() {
  const imgFrameContainer = document.getElementById("imgFrame");
  const navButtons = imgFrameContainer.querySelectorAll("button");
  let leftNavButton = navButtons[0];
  let rightNavButton = navButtons[1];

  leftNavButton.addEventListener("click", async () => {
    let newIndex = Math.max(imgSelector.selectedIndex - 1, 0);
    if (newIndex >= 0) {
      imgSelector.selectedIndex = newIndex;
      refreshImageOfMain();
      refreshFrameDetectStatus();
    }
  });

  rightNavButton.addEventListener("click", async () => {
    let maxIndex = imgSelector.childElementCount;
    let newIndex = Math.min(imgSelector.selectedIndex + 1, maxIndex);
    if (newIndex < maxIndex) {
      imgSelector.selectedIndex = newIndex;
      refreshImageOfMain();
      refreshFrameDetectStatus();
    }
  });
}

async function refreshImageOfMain() {
  var img = document.getElementById("imageContainer");
  const selectedValue = imgSelector.value;
  img.src = `${urlImagesStoreDir}/${selectedValue}`;
  img.alt = `${selectedValue}`;
}

async function refreshFrameDetectStatus() {
  // refreshFrameDetectStatus
  const selectedValue = imgSelector.value;

  const response = await fetch(
    `/api/get_state_for_delivery_by_name/${selectedValue}`,
    {
      method: "GET",
      headers: { Accept: "application/json" },
    }
  );
  // если запрос прошел нормально
  if (response.ok) {
    // получаем данные
    const deliveryItemData = await response.json();
    document.getElementById("table_12").textContent =
      deliveryItemData.founded_screw_flat;
    document.getElementById("table_22").textContent =
      deliveryItemData.founded_screw_plus;
    document.getElementById("table_32").textContent =
      deliveryItemData.founded_offset_plus_screw;
    document.getElementById("table_42").textContent =
      deliveryItemData.founded_kolovorot;
    document.getElementById("table_52").textContent =
      deliveryItemData.founded_safety_pliers;
    document.getElementById("table_62").textContent =
      deliveryItemData.founded_pliers;
    document.getElementById("table_72").textContent =
      deliveryItemData.founded_shernitsa;
    document.getElementById("table_82").textContent =
      deliveryItemData.founded_adjustable_wrench;
    document.getElementById("table_92").textContent =
      deliveryItemData.founded_can_opener;
    document.getElementById("table_102").textContent =
      deliveryItemData.founded_open_end_wrench;
    document.getElementById("table_112").textContent =
      deliveryItemData.founded_side_cutters;
    console.log(deliveryItemData);
  }
}

imgSelector.addEventListener("change", () => {
  refreshImageOfMain();
  refreshFrameDetectStatus();
});

async function massUploadFiles(containerInputImages) {
  containerScreen1.classList.add("hidden");
  containerScreen2.classList.add("hidden");
  containerLoading.classList.remove("hidden");
  const files = containerInputImages.files;
  if (!files) return;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    console.log(`Try to upload file ${files[i]}`);
    formData.append("files", files[i]);
  }

  try {
    const response = await fetch("/upload-multiple-files-advanced/", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const result = await response.json();
      console.log("Успешная загрузка:", result.message);
      alert(`Файлы успешно загружены! 
              ${result.message}`);
      refreshFileSelector();
      refreshImageOfMain();
      containerScreen2.classList.remove("hidden");
      containerLoading.classList.add("hidden");
    } else {
      const result = await response.json();
      console.error("Ошибка загрузки:", response.statusText);
      alert(`Ошибка при загрузке файлов. 
          ${result.detail}${result.msg}`);
      containerScreen1.classList.remove("hidden");
      containerScreen2.classList.add("hidden");
      containerLoading.classList.add("hidden");
    }
  } catch (error) {
    console.error("Ошибка при отправке запроса:", error);
    alert("Ошибка при отправке запроса на сервер.");

    containerScreen1.classList.remove("hidden");
    containerScreen2.classList.add("hidden");
    containerLoading.classList.add("hidden");
  }
}

containerInputScreen1.addEventListener("change", () => {
  massUploadFiles(containerInputScreen1);
});
containerInputScreen2.addEventListener("change", () => {
  massUploadFiles(containerInputScreen2);
});

containerSwithShowRawImage.addEventListener("change", function () {
  if (this.checked) {
    urlImagesStoreDir = "http://localhost:8000/results";
    console.log("Отображаем с боксами");
    refreshImageOfMain();
  } else {
    urlImagesStoreDir = "http://localhost:8000/raw_images";
    console.log("Отображаем без боксов");
    refreshImageOfMain();
  }
});

containerClearDbAndStorage.addEventListener("click", async () => {
  console.log("clean storage");
  containerLoading.classList.remove("hidden");
  containerScreen2.classList.add("hidden");

  const formData = new FormData();
  try {
    const response = await fetch("/api/clear_database_and_storage/", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const result = await response.json();
      console.log("Файлы успешно удалены!", result.message);
      alert(`Файлы успешно удалены! 
              ${result.message}`);
      refreshFileSelector();
      refreshImageOfMain();
    } else {
      const result = await response.json();
      console.error("Ошибка удаления файлов.", response.statusText);
      alert(`Ошибка удаления файлов.
          ${result.detail}`);
    }
  } catch (error) {
    console.error("Ошибка при отправке запроса:", error);
    alert("Ошибка при отправке запроса на сервер.");
  }

  containerLoading.classList.add("hidden");
  containerScreen1.classList.remove("hidden");
  refreshFileSelector();
  refreshImageOfMain();
});

refreshFileSelector();
addMoveBtns();

// if image
containerLoading.classList.add("hidden");
containerScreen1.classList.add("hidden");
containerScreen2.classList.add("hidden");
refreshImageOfMain();

if (imgSelector.length !== 0) {
  // containerScreen1.classList.add('hidden');
  imgSelector.selectedIndex = 0;

  containerScreen2.classList.remove("hidden");

  refreshImageOfMain();
}

console.log(`imgSelector after ${imgSelector.value}`);

if (document.readyState === "complete") {
  // Your code to execute when the page is fully loaded
  console.log("Page is complete!");
  refreshImageOfMain();
  // console.log(`test ${document.getElementById('currentImageFileSelector').value}`) ;
} else {
  window.addEventListener("load", () => {
    // Your code to execute when the page is fully loaded
    console.log("Page is complete!");
    refreshImageOfMain();
    // console.log(`test ${document.getElementById('currentImageFileSelector').value}`) ;
  });
}
