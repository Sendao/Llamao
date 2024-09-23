//const { contextBridge, ipcMain } = require('electron');
/*
window.elect.receive('gotImage', function(path) {
    console.log("local gotImage: ", path);
    window.gotImage(path);
});
*/

function generate() {
    window.elect.runQuery(window, document.forms[0].data.value);
}
