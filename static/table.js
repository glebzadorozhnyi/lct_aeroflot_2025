let pressedKeys = new Set();

const resultImg = document.getElementById("resultImg");
const image_window = document.getElementById("imageContainer");
// import { refreshImageOfMain, refreshFileSelector } from '/static/main.js'

function updateFrameCounter() {
  frameCounter = document.getElementById("frameCounter");
  console.log(imgSelector.childElementCount, imgSelector.selectedIndex);
  frameCounter.innerText =
    imgSelector.selectedIndex + 1 + "/" + imgSelector.childElementCount;
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

image_window.addEventListener("load", () => {
  setTimeout(() => {
    highlightDifferences();
  }, 100);
  updateFrameCounter();
});

document.addEventListener("keydown", function (event) {
  if (
    event.key === "ArrowLeft" ||
    event.key === "ArrowUp"
  ) {
  const imgSelector = document.getElementById("currentImageFileSelector");
    let newIndex = imgSelector.selectedIndex - 1;
    if (newIndex >= 0) {
      imgSelector.selectedIndex = newIndex;
    } else {
      imgSelector.selectedIndex = imgSelector.childElementCount - 1;
    }
    refreshImageOfMain();
    refreshFrameDetectStatus();
  }
  else if (
    event.key === "ArrowRight" ||
    event.key === "ArrowDown"
  )
  {
    let maxIndex = imgSelector.childElementCount;
    let newIndex = Math.min(imgSelector.selectedIndex + 1, maxIndex);
    if (newIndex < maxIndex) {
      imgSelector.selectedIndex = newIndex;
    } else {
      imgSelector.selectedIndex = 0;
    }
    refreshImageOfMain();
    refreshFrameDetectStatus();
  }
});

document.addEventListener("DOMContentLoaded", function () {
  // Находим все кнопки "+" в корзине
  const plusButtons = document.querySelectorAll(".cart-list .qty-btn.plus");

  // Находим все кнопки "-" в корзине
  const minusButtons = document.querySelectorAll(".cart-list .qty-btn.minus");

  plusButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
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

      // Триггерим событие change, если есть обработчики пересчета
      input.dispatchEvent(new Event("change"));

      console.log(`Количество товара ${row.dataset.id} увеличено до ${qty}`);
    });
  });
  highlightDifferences();

  // document.getElementById("StartScreen").addEventListener("change", function() {
  //     const screen1 = document.getElementById("screen1");
  //     const screen2 = document.getElementById("screen2");
  //     if (this.checked) {
  //           screen1.classList.remove("hidden");
  //           screen2.classList.add("hidden");

  //     } else {
  //           screen1.classList.add("hidden");
  //           screen2.classList.remove("hidden");
  //     }
  //   });

  minusButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
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
      if (qty < 0) qty = 0;
      input.value = qty;

      // Триггерим событие change, если есть обработчики пересчета
      input.dispatchEvent(new Event("change"));

      console.log(`Количество товара ${row.dataset.id} увеличено до ${qty}`);
    });
  });
  highlightDifferences();
});
