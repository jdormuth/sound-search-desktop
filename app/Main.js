const electron = require('electron')
// // Module to control application life.
const app = electron.app

//for communiation between dom and main
var ipc = require('electron').ipcMain;


console.log(require.resolve('electron'));
// Module to create native browser window.
const BrowserWindow = electron.BrowserWindow

const path = require('path')
const url = require('url')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow

function createWindow () {
  // Create the browser window.
  mainWindow = new BrowserWindow({width: 800, height: 600})

  // and load the index.html of the app.
  mainWindow.loadURL(url.format({
    pathname: '/Users/jacob/sound-search-desktop/index.html',
    protocol: 'file:',
    slashes: true
  }))

  // Open the DevTools.
  mainWindow.webContents.openDevTools()

  // Emitted when the window is closed.
  mainWindow.on('closed', function () {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', function () {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) {
    createWindow()
  }
})
var Drums = require("./Drums");


  
  var drums = new Drums();
  //var cover = document.getElementById("cover");
  
  //when start button pressed
  ipc.on('startAction', function(event, data){
    
    var intro = data;
    //data.style.display = "none";
    event.sender.send('startReply', '');
  });
  var trackList = [];
  ipc.on('tsne', function(event, data){
    var args = data;
    fileNames = args[0];
    userFile = args[1];
    var spawn = require("child_process").spawn;
    var process = spawn('python',["./scripts/collect-samples.py",fileNames,userFile]);

    process.stdout.on('data', function (data){
      // Do something with the data returned from python script
      if(data.toString().replace(/\s+/g, '') == "listen"){
        console.log("new results");
        trackList = [];
      }
      if(data.toString()[0] == '/'){
        trackList.push(data.toString());
      }
      var data = data.toString().replace(/\s+/g, '');
      console.log(data);
      console.log(data.length);
      console.log(data== "finished");

      if(data === "finished"){
        console.log("main finished");
        event.sender.send("tsneFinished",trackList);
      }
      
      
      
    });
  });


  //whenever a button is pressed
  ipc.on("first", function(event, data){
    event.sender.send("firstButtonResponse",trackList[0]);
  });
  ipc.on("second", function(event,data){

  });
  ipc.on("third", function(event, data){
    console.log(trackList[4]);
  });
  ipc.on("fourth", function(event, data){
    console.log(trackList[4]);
  });