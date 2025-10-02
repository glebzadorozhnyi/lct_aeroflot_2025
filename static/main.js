

const imgSelector = document.getElementById('currentImageFileSelector');
const containerInputScreen2 = document.getElementById('fileInputScreen2');
const containerInputScreen1 = document.getElementById('fileInputScreen1');

const containerSwithShowRawImage = document.getElementById('switchShowRawImage');
const containerClearDbAndStorage = document.getElementById('BtnCleanerDatabaseAndImageStorage');
const containerLoading = document.getElementById('loading');
const containerScreen2 = document.getElementById('screen2');
const containerScreen1 = document.getElementById('screen1');
const containerCartListItemsPanel = document.getElementById('cartListItemsPanel');



const cart_items = [
  {"id": 1, "name": "Плоская отвёртка"},
  {"id": 2, "name": "Крестовая отвёртка"},
  {"id": 3, "name": "Отвёртка крест"},
  {"id": 4, "name": "Коловорот"},
  {"id": 5, "name": "Пассатижи контр"},
  {"id": 6, "name": "Пассатижи"},
  {"id": 7, "name": "Шерница"},
  {"id": 8, "name": "Разводной ключ"},
  {"id": 9, "name": "Открывалка"},
  {"id": 10, "name": "Ключ рожковый"},
  {"id": 11, "name": "Бокорезы"},
]


let urlImagesStoreDir = "http://localhost:8000/results";


console.log(`imgSelector before ${imgSelector.value}`);


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
  // containerLoading.classList.remove('.is_hidden');
  // containerLoading.style.display = 'block';
  // resultElement.textContent = '';
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
  // containerLoading.style.display = 'none';
  // containerLoading.classList.add('.is_hidden');
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
  containerLoading.classList.remove('hidden');
  containerScreen2.classList.add('hidden');
  const response = await fetch("/api/get_all_uploaded_files", {

    method: "GET",
    
    headers: { "Accept": "application/json" }
  });
  if (response.ok){
    const countOfUploadedFiles = await response.json();
    if (countOfUploadedFiles.message !== "base_is_empty"){
      imgSelector.innerHTML = '';
      countOfUploadedFiles.forEach(fileName => imgSelector.append(genHtmlRawOption(fileName)));
      containerScreen2.classList.remove('hidden');
      containerLoading.classList.add('hidden');
    }
    else{
      containerScreen1.classList.remove('hidden');
      containerScreen2.classList.add('hidden');
      containerLoading.classList.add('hidden');  
    }
  }
  else{
    containerScreen1.classList.remove('hidden');
    containerScreen2.classList.add('hidden');
    containerLoading.classList.add('hidden');
  }


  
}

function addMoveBtns(imgFrameContainer){

  // let currentIndex = imgSelector.selectedIndex;
  let newIndex;
  let maxIndex = imgSelector.childElementCount;

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
    newIndex = Math.max(imgSelector.selectedIndex - 1, 0);
    console.log(`Press left ${newIndex}`)
    if (newIndex >= 0) {
      imgSelector.selectedIndex = newIndex;
      console.log(`selectedIndex ${imgSelector.selectedIndex}`);
      refreshImageOfMain();
      refreshFrameDetectStatus();
    }

 
  });
  
  const rightBtn = document.createElement('button');
  rightBtn.className = 'nav-btn right';
  imgFrameContainer.appendChild(rightBtn);

  rightBtn.addEventListener('click', async () => {

    newIndex = Math.min(imgSelector.selectedIndex+1 , maxIndex);
    console.log(`Press Right ${newIndex}`)
    console.log(`imgSelector.selectedIndex ${imgSelector.selectedIndex}`)
    console.log(`maxIndex ${maxIndex}`);
    if (newIndex < maxIndex) {
      imgSelector.selectedIndex = newIndex;
      console.log(`selectedIndex ${imgSelector.selectedIndex}`);
      refreshImageOfMain();
      refreshFrameDetectStatus();
    }
  });
}

async function refreshImageOfMain() {
  
  const imgFrameContainer = document.getElementById('imgFrame'); // Ваш селектор (например, <button>)
  
  console.log(`select image ${imgSelector.value}`);

  const  selectedValue = imgSelector.value;


  imgFrameContainer.innerHTML = '';
  if (selectedValue) {
    const img = document.createElement('img');
    img.src = `${urlImagesStoreDir}/${selectedValue}`;
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

// document.getElementById("fileInput").addEventListener("change", async () => {
//   refreshFileSelector();
// });

imgSelector.addEventListener('change', () => {
  refreshImageOfMain();
  refreshFrameDetectStatus();
});



async function massUploadFiles(containerInputImages) {
  containerScreen1.classList.add('hidden');
  containerScreen2.classList.add('hidden');
  containerLoading.classList.remove('hidden');
  const files = containerInputImages.files;
  if (!files) return;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    console.log(`Try to upload file ${files[i]}`)
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
          containerScreen2.classList.remove('hidden');
          containerLoading.classList.add('hidden');
      } else {
          const result = await response.json();
          console.error('Ошибка загрузки:', response.statusText);
          alert(`Ошибка при загрузке файлов. 
          ${result.detail}${result.msg}`);
          containerScreen1.classList.remove('hidden');
          containerScreen2.classList.add('hidden');
          containerLoading.classList.add('hidden');
      }
  } catch (error) {
      console.error('Ошибка при отправке запроса:', error);
      alert('Ошибка при отправке запроса на сервер.');
      
      containerScreen1.classList.remove('hidden');
      containerScreen2.classList.add('hidden');
      containerLoading.classList.add('hidden');
  }

}

containerInputScreen1.addEventListener('change', () => {
  massUploadFiles(containerInputScreen1);
});
containerInputScreen2.addEventListener('change', () => {
  massUploadFiles(containerInputScreen2);
});

containerSwithShowRawImage.addEventListener("change", function() {
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


containerClearDbAndStorage.addEventListener('click', async () => {
  console.log("clean storage");
  containerLoading.classList.remove('hidden');
  containerScreen2.classList.add('hidden');

  const formData = new FormData();
  try {
      const response = await fetch('/api/clear_database_and_storage/', {
          method: 'POST',
          body: formData
      });

      if (response.ok) {
          const result = await response.json();
          console.log('Файлы успешно удалены!', result.message);
          alert(`Файлы успешно удалены! 
              ${result.message}`);
          refreshFileSelector();
          refreshImageOfMain();
      } else {
          const result = await response.json();
          console.error('Ошибка удаления файлов.', response.statusText);
          alert(`Ошибка удаления файлов.
          ${result.detail}`);
      }
  } catch (error) {
      console.error('Ошибка при отправке запроса:', error);
      alert('Ошибка при отправке запроса на сервер.');
  }
  
  containerLoading.classList.add('hidden');
  containerScreen1.classList.remove('hidden');
  refreshFileSelector();
  refreshImageOfMain();
});



async function renderCartListItemsPanel(containerInputImages) {
  console.log(`Element at index TEST`);
  cart_items.forEach(function(item, index) {
    console.log(`Create element at index ${index}: ${item.id} ${item.name}`);
    const i_divRaw = document.createElement('div');
    // i_divRaw.textContent = item.name;
    
    // i_divRaw.textContent = item.name;
    i_divRaw.innerHTML =`
    <div id="row_${index}" class="row-name">${item.name}</div>

    <div class= "cart-row" style="display: flex; align-items: center;">
      <button class="qty-btn minus" type="button" aria-label="Уменьшить">−</button>
      <input id="table_${index}1" class="qty-input" type="text" inputmode="numeric" pattern="[0-9]*" value="1" aria-label="Количество">
      <button class="qty-btn plus" type="button" aria-label="Увеличить">+</button>
    </div>

    <div style="text-align: center;">
      <span id="table_${index}2" class="qty-text"></span>
    </div>`

    containerCartListItemsPanel.appendChild(i_divRaw);
    

  });
}

// async function renderCartListItemsPanel(){
  // cartListItemsPanel
//   const numbers = [1, 2, 3, 4, 5];
//   console.log(`Element at index`);
  
//   numbers.forEach(function(number, index) {
//       console.log(`Element at index ${index}: ${number}`);
//   });
//   cart_items.forEach(function(number, index) {
//     console.log(`Element at index ${index}: ${number}`);
// });
// // }



// renderCartListItemsPanel();
refreshFileSelector();

// if image
containerLoading.classList.add('hidden');
containerScreen1.classList.add('hidden');
containerScreen2.classList.add('hidden');
refreshImageOfMain();

if (imgSelector.length !== 0) {
  // containerScreen1.classList.add('hidden');
  imgSelector.selectedIndex = 0;

  containerScreen2.classList.remove('hidden');

  refreshImageOfMain();
}


function highlightDifferences() {
  const status_box = document.getElementById(`checkStatus`);
  let allMatch = true;
      for (let i = 1; i <= 11; i++) {
          const input = document.getElementById(`table_${i}1`);
          const span = document.getElementById(`table_${i}2`);
          const row = document.getElementById(`row_${i}`);

          if (!input || !span || !row) continue; // на всякий случай

          if (input.value != span.textContent) {
              row.style.backgroundColor = "#ffcccc"; // красный фон
              allMatch = false;
          } else {
              row.style.backgroundColor = "#ffffff"; // белый фон
          }
      }
          if (allMatch) {
              status_box.textContent = "Количество инструментов соответствует ожидаемому";
              status_box.style.color = "green";
          } else {
              status_box.textContent = "Количество инструментов не совпадает с ожидаемым";
              status_box.style.color = "red";
          }
}





function renderRightCartList()
{
  // Находим все кнопки "+" в корзине
  const plusButtons = document.querySelectorAll(".cart-list .qty-btn.plus");

  // Находим все кнопки "-" в корзине
  const minusButtons = document.querySelectorAll(".cart-list .qty-btn.minus");

  plusButtons.forEach(btn => {
    btn.addEventListener("click", function() {
      // Находим родительский ряд
      const row = btn.closest(".cart-row");
      if (!row) return;

      // Находим input с количеством в этом ряду
      const input = row.querySelector(".qty-input");
      // Точно так же можно находить текст const text = row.querySelector(".qty-text");
      if (!input) return;

      // Получаем текущее значение, увеличиваем на 1
      let qty = parseInt(input.value) || 0;
      qty += 1;
      input.value = qty;
      highlightDifferences();

      // Триггерим событие change, если есть обработчики пересчета
      input.dispatchEvent(new Event('change'));

      console.log(`Количество товара ${row.dataset.id} увеличено до ${qty}`);
    });
  });

  minusButtons.forEach(btn => {
    btn.addEventListener("click", function() {
      // Находим родительский ряд
      const row = btn.closest(".cart-row");
      if (!row) return;

      // Находим input с количеством в этом ряду
      const input = row.querySelector(".qty-input");
      // Точно так же можно находить текст const text = row.querySelector(".qty-text");
      if (!input) return;


      // Получаем текущее значение, увеличиваем на 1
      let qty = parseInt(input.value) || 0;
      qty -= 1;
      if (qty < 0) qty = 0
      input.value = qty;
      highlightDifferences();

      // Триггерим событие change, если есть обработчики пересчета
      input.dispatchEvent(new Event('change'));

      console.log(`Количество товара ${row.dataset.id} увеличено до ${qty}`);
    });
  });
}


document.addEventListener("DOMContentLoaded", renderRightCartList() );


renderCartListItemsPanel();

document.addEventListener('DOMContentLoaded', () => {
  // Your code to execute after the DOM is ready
  console.log('DOM is fully loaded and parsed!');
  // refreshImageOfMain();
  // console.log(`test ${document.getElementById('currentImageFileSelector').value}`) ;
});


// var imgSelector = document.getElementById('currentImageFileSelector');

console.log(`imgSelector after ${imgSelector.value}`);

if (document.readyState === 'complete') {
  // Your code to execute when the page is fully loaded
  console.log('Page is complete!');
  refreshImageOfMain();
  // console.log(`test ${document.getElementById('currentImageFileSelector').value}`) ;
} else {
  window.addEventListener('load', () => {
    // Your code to execute when the page is fully loaded
    console.log('Page is complete!');
    refreshImageOfMain();
    // console.log(`test ${document.getElementById('currentImageFileSelector').value}`) ;
  });
}

