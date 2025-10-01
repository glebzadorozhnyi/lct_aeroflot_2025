

const imgSelector = document.getElementById('currentImageFileSelector');
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
  renewDeliveryTable();
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
  renewDeliveryTable();
}


async function editTool(aeroTool_id){
  console.log(`editTool ${aeroTool_id}`);
}
async function deleteTool(aeroTool_id){
  console.log(`deleteTool ${aeroTool_id}`);
}


async function genHtmlRawOfDeliveryBase(aeroToolDelivery) {

  // console.log(`${aeroToolDelivery}`);
  console.log(aeroToolDelivery.image_file_id);



  const tr = document.createElement("tr");
  // tr.setAttribute("data-rowid", aeroToolDelivery.id);
  tr.setAttribute("data-rowid", "1");

  const nameTd = document.createElement("td");
  nameTd.append(aeroToolDelivery.image_file_id);
  tr.appendChild(nameTd);

  const typeTd = document.createElement("td");
  typeTd.append(aeroToolDelivery.id);
  tr.append(typeTd);

  const detectStateTd = document.createElement("td");
  detectStateTd.append(aeroToolDelivery.id);
  tr.append(detectStateTd);

  const linksTd = document.createElement("td");

  // const setDetectStateLink = document.createElement("button"); 
  // setDetectStateLink.append("Set \"detected\"");
  // setDetectStateLink.addEventListener("click", async() => await setDetectState(aeroTool.id));
  // linksTd.append(setDetectStateLink);

  // const unsetDetectStateLink = document.createElement("button"); 
  // unsetDetectStateLink.append("Unset \"detected\"");
  // unsetDetectStateLink.addEventListener("click", async () => await unsetDetectState(aeroTool.id));
  // linksTd.append(unsetDetectStateLink);

  const editLink = document.createElement("button"); 
  // editLink.append("Edit");
  // editLink.addEventListener("click", async() => await editTool(aeroTool.id));
  linksTd.append(editLink);

  const removeLink = document.createElement("button"); 
  removeLink.append("Delete");
  linksTd.append(removeLink);

  tr.appendChild(linksTd);

  return tr;
}




function getTdWithAttr(attr){
  // const td=document.createAttribute

  const td = document.createElement('td');
  td.textContent = attr;
  return td;
  
   
}

async function renewDeliveryTable() {
    resetToolTable();
  // отправляет запрос и получаем ответ
  const response = await fetch("/api/get_all_deliveries", {
      method: "GET",
      headers: { "Accept": "application/json" }
  });

  const container = document.getElementById('tableContainer');

  // если запрос прошел нормально
  if (response.ok === true) {
      const jsonData = await response.json();

      const table = document.createElement('table');
      const thead = document.createElement('thead');
      // jsonDa ta.forEach(item => {
        const headerRow = document.createElement('tr');
        // const headers = Object.keys(item);
        const headers = ["id", "upload_time", 
          "screw_flat",  // 1. Плоская отвертка (-)
          "screw_plus",  // 2. Крестовая отвертка (+)
          "offset_plus_screw",  // 3. отвертка на смещенный крест
          "kolovorot",  // 4. Коловорот
          "safety_pliers",  // 5. Пассатижи контровочные
          "pliers",  // 6. Пассатижи
          "shernitsa",  // 7. Шерница
          "adjustable_wrench",  // 8. Разводной ключ
          "can_opener",  // 9. Открывалка для банок с маслом
          "open_end_wrench",  // 10. Ключ рожковый накидной 3/4
          "side_cutters",  // 11. Бокорезы
      
        ];
        headers.forEach(headerText => {
            const th = document.createElement('th');
            th.textContent = headerText;
            headerRow.appendChild(th);
        });

        
        thead.appendChild(headerRow);


      jsonData.forEach(item => {
        
        console.log(item.id);
        const tbody = document.createElement('tbody');

        table.appendChild(thead);  
          // Create data row
          const dataRow = document.createElement('tr');
          dataRow.appendChild(getTdWithAttr(item.id));
          dataRow.appendChild(getTdWithAttr(item.datatime));
          dataRow.appendChild(getTdWithAttr(item.founded_screw_flat));
          dataRow.appendChild(getTdWithAttr(item.founded_screw_plus));
          dataRow.appendChild(getTdWithAttr(item.founded_offset_plus_screw));
          dataRow.appendChild(getTdWithAttr(item.founded_kolovorot));
          dataRow.appendChild(getTdWithAttr(item.founded_safety_pliers));
          dataRow.appendChild(getTdWithAttr(item.founded_pliers));
          dataRow.appendChild(getTdWithAttr(item.founded_shernitsa));
          dataRow.appendChild(getTdWithAttr(item.founded_adjustable_wrench));
          dataRow.appendChild(getTdWithAttr(item.founded_can_opener));
          dataRow.appendChild(getTdWithAttr(item.founded_open_end_wrench));
          dataRow.appendChild(getTdWithAttr(item.founded_side_cutters));   
          

          tbody.appendChild(dataRow);
          table.appendChild(tbody);
    
          container.appendChild(table);
          container.appendChild(document.createElement('br')); // Add a line break for separation
      });
  

  }
}

document.getElementById("showAeroToolsTable").addEventListener("click", async () => {
  console.log(`Hello! Init table...`);
  renewDeliveryTable();

});

function genHtmlRawOption(fileName) {
  
  const option = document.createElement("option");
  const filename_without_hash = fileName.slice(33);
  option.setAttribute("value", fileName);
  option.text = filename_without_hash;
  option.className = "image-selector-option";
  option.text = filename_without_hash;
  option.className = "image-selector-option";
  return option;
}

async function refreshFileSelector() {
  
  const response = await fetch("/api/get_all_uploaded_files", {
    method: "GET",
    
    headers: { "Accept": "application/json" }
  });
  const countOfUploadedFiles = await response.json();

  countOfUploadedFiles.forEach(fileName => imgSelector.append(genHtmlRawOption(fileName)));
}

function addMoveBtns(imgFrameContainer){

    function handleKeys() {
          if (pressedKeys.has('о') || pressedKeys.has('j')) {
              show_picture(-1);
          }

          if (pressedKeys.has('л') || pressedKeys.has('k')) {
              show_picture(1);
          }
  }


  const leftBtn = document.createElement('button');
  leftBtn.className = 'nav-btn left';
  imgFrameContainer.appendChild(leftBtn);

  leftBtn.addEventListener('click', async () => {
    // imgSelector
    console.log('Press left')
  });
  


  const rightBtn = document.createElement('button');
  rightBtn.className = 'nav-btn right';
  imgFrameContainer.appendChild(rightBtn);

  rightBtn.addEventListener('click', async () => {
    // imgSelector
    console.log('Press Right')
  });
}

async function refreshImageOfMain() {
  
  const imgFrameContainer = document.getElementById('imgFrame'); // Ваш селектор (например, <button>)
  const selectedValue = imgSelector.value;
  imgFrameContainer.innerHTML = '';
  if (selectedValue) {
    const img = document.createElement('img');
    img.src = `http://localhost:8000/results/${selectedValue}`;
    img.alt = `${selectedValue}`;

    addMoveBtns(imgFrameContainer);
    


    
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
