
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
  // sayHello('World');

  console.log(`Hello! Init table...`);
  getAeroTools();

});