

const imgSelector = document.getElementById('mySelect');
const containerInputImages = document.getElementById('fileInput');

function resetToolTable() {
  
  const parentElement = document.querySelector('tbody');
  console.log("Found existing table");
  if (parentElement) {
    console.log("Clean tool table");
    parentElement.innerHTML = '';
  }

}


async function editAeroTool(aeroToolId, aeroToolName, aeroToolAge) {
  const response = await fetch("api/aeroTools", {
      method: "PUT",
      headers: { "Accept": "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({
          id: aeroToolId,
          name: aeroToolName,
          age: parseInt(aeroToolAge, 10)
      })
  });
  if (response.ok === true) {
      const aeroTool = await response.json();
      document.querySelector(`tr[data-rowid='${aeroTool.id}']`).replaceWith(row(aeroTool));
  }
  else {
      const error = await response.json();
      console.log(error.message);
  }
}


async function setDetectState(aeroTool_id){
  console.log(`setDetectState ${aeroTool_id}`);
  const response = await fetch(`/api/set_detect_state/${aeroTool_id}`, {
      method: "POST",
      headers: { "Accept": "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({
          id: aeroTool_id,
          // name: aeroToolName,
      })
  });
  if (response.ok === true) {
      const aeroTool = await response.json();
  }
  else {
      const error = await response.json();
      console.log(error.message);
  }
  getAeroTools();
}

async function unsetDetectState(aeroTool_id){
  console.log(`unsetDetectState ${aeroTool_id}`);
  const response = await fetch(`/api/unset_detect_state/${aeroTool_id}`, {
      method: "POST",
      headers: { "Accept": "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({
          id: aeroTool_id,
          // name: aeroToolName,
      })
  });
  if (response.ok === true) {
      const aeroTool = await response.json();
  }
  else {
      const error = await response.json();
      console.log(error.message);
  }
  getAeroTools();
}


async function editTool(aeroTool_id){
  console.log(`editTool ${aeroTool_id}`);
}
async function deleteTool(aeroTool_id){
  console.log(`deleteTool ${aeroTool_id}`);
}


function genHtmlRawOfTool(aeroTool) {
  
  const tr = document.createElement("tr");
  tr.setAttribute("data-rowid", aeroTool.id);

  const nameTd = document.createElement("td");
  nameTd.append(aeroTool.name);
  tr.append(nameTd);

  const typeTd = document.createElement("td");
  typeTd.append(aeroTool.type);
  tr.append(typeTd);

  const detectStateTd = document.createElement("td");
  detectStateTd.append(aeroTool.detect_state);
  tr.append(detectStateTd);

  const linksTd = document.createElement("td");

  const setDetectStateLink = document.createElement("button"); 
  setDetectStateLink.append("Set \"detected\"");
  setDetectStateLink.addEventListener("click", async() => await setDetectState(aeroTool.id));
  linksTd.append(setDetectStateLink);

  const unsetDetectStateLink = document.createElement("button"); 
  unsetDetectStateLink.append("Unset \"detected\"");
  unsetDetectStateLink.addEventListener("click", async () => await unsetDetectState(aeroTool.id));
  linksTd.append(unsetDetectStateLink);

  // const editLink = document.createElement("button"); 
  // editLink.append("Edit");
  // editLink.addEventListener("click", async() => await editTool(aeroTool.id));
  // linksTd.append(editLink);

  // const removeLink = document.createElement("button"); 
  // removeLink.append("Delete");
  // removeLink.addEventListener("click", async () => await deleteTool(aeroTool.id));
  // linksTd.append(removeLink);

  tr.appendChild(linksTd);

  return tr;
}




async function getAeroTools() {
  resetToolTable();
  // отправляет запрос и получаем ответ
  const response = await fetch("/api/get_state_for_delivery/1", {
      method: "GET",
      headers: { "Accept": "application/json" }
  });
  // если запрос прошел нормально
  if (response.ok === true) {
      // получаем данные
      const aeroTools = await response.json();
      const rows = document.querySelector("tbody");
      // добавляем полученные элементы в таблицу
      aeroTools.forEach(aeroTool => rows.append(genHtmlRawOfTool(aeroTool)));
  }
}

document.getElementById("showAeroToolsTable").addEventListener("click", async () => {
  console.log(`Hello! Init table...`);
  getAeroTools();

});


function genHtmlRawOption(fileName) {
  
  const option = document.createElement("option");
  option.setAttribute("value", fileName);
  option.text = fileName;
  return option;
}


async function refreshFileSelector() {
  
  const response = await fetch("/get_all_uploaded_files", {
    method: "GET",
    
    headers: { "Accept": "application/json" }
  });
  const countOfUploadedFiles = await response.json();

  const imgSelector = document.getElementById("mySelect");

  countOfUploadedFiles.forEach(fileName => imgSelector.append(genHtmlRawOption(fileName)));
}


async function refreshImageOfMain() {
  
  const imgFrameContainer = document.getElementById('imgFrame'); // Ваш селектор (например, <button>)
  const selectedValue = imgSelector.value;
  imgFrameContainer.innerHTML = '';
  if (selectedValue) {
    const img = document.createElement('img');
    img.src = `http://localhost:8000/results/${selectedValue}`;
    img.alt = `${selectedValue}`;
    
    // Optional: Add loading indicator
    img.style.display = 'none';
    imgFrameContainer.appendChild(img);
    
    // Handle image load
    img.onload = () => {
        img.style.display = 'block';
    };
    
    // Handle load errors
    img.onerror = () => {
      imgFrameContainer.innerHTML = '<p style="color: red;">Failed to load image</p>';
    };
  } else {
      // Show placeholder when no selection
      imgFrameContainer.innerHTML = '<span class="placeholder">Select an image from the dropdown</span>';
  }
}

async function refreshFrameDetectStatus() {
  // refreshFrameDetectStatus
  const selectedValue = imgSelector.value;

  const response = await fetch(`/api/get_state_for_delivery_by_name/${selectedValue}`, {
    method: "GET",
    headers: { "Accept": "application/json" }
  });
  // если запрос прошел нормально
  if (response.ok === true) {
      // получаем данные
      const deliveryItemData = await response.json();
      document.getElementById('table_12').textContent =deliveryItemData.founded_screw_flat;
      document.getElementById('table_22').textContent =deliveryItemData.founded_screw_plus;
      document.getElementById('table_32').textContent =deliveryItemData.founded_offset_plus_screw;
      document.getElementById('table_42').textContent =deliveryItemData.founded_kolovorot;
      document.getElementById('table_52').textContent =deliveryItemData.founded_safety_pliers;
      document.getElementById('table_62').textContent =deliveryItemData.founded_pliers;
      document.getElementById('table_72').textContent =deliveryItemData.founded_shernitsa;
      document.getElementById('table_82').textContent =deliveryItemData.founded_adjustable_wrench;
      document.getElementById('table_92').textContent =deliveryItemData.founded_can_opener;
      document.getElementById('table_102').textContent =deliveryItemData.founded_open_end_wrench;
      document.getElementById('table_112').textContent =deliveryItemData.founded_side_cutters ;    
      console.log(deliveryItemData);
  }

}

document.getElementById("fileInput").addEventListener("change", async () => {
  refreshFileSelector();
});

imgSelector.addEventListener('change', () => {
  refreshImageOfMain();
  refreshFrameDetectStatus();
});

containerInputImages.addEventListener('change', async () => {
  const files = containerInputImages.files;
  if (!files) return;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
  }


  try {
      const response = await fetch('/upload-multiple-files-advanced/', {
          method: 'POST',
          body: formData
      });

      if (response.ok) {
          const result = await response.json();
          console.log('Успешная загрузка:', result.message);
          alert(`Файлы успешно загружены! 
              ${result.message}`);
          refreshFileSelector();
          refreshImageOfMain();
      } else {
          const result = await response.json();
          console.error('Ошибка загрузки:', response.statusText);
          alert(`Ошибка при загрузке файлов. 
          ${result.detail}`);
      }
  } catch (error) {
      console.error('Ошибка при отправке запроса:', error);
      alert('Ошибка при отправке запроса на сервер.');
  }

});

refreshFileSelector();
refreshImageOfMain();
