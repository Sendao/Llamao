const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('elect', {
    runQuery: (wind, data) => { ipcRenderer.invoke('runquery', wind, data); console.log("Sent ", wind, data); },
    interface: (wind, data) => { ipcRenderer.invoke('interface', wind, data); console.log("XF ", wind, data); },
    receive: (ch, fn) => {
        ipcRenderer.on(ch, (event, ...args) => fn(...args));
    }
});
