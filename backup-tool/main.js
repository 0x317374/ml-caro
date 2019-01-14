const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const modelsDir = path.normalize(process.argv[2]);
const serverUrl = process.argv[3];

const wait = time => new Promise(rel => setTimeout(rel, time));

const backup = () => {
  const time = new Date().toLocaleString('vi-VN', { year: '2-digit', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  console.log(`[${time}] Backup started:`);
  
  fs.readdir(modelsDir, async (error, files) => {
    if (error) return console.log(error);
    
    const modelFiles = files.filter(file => file.startsWith('keras') && file.endsWith('.model'));
    
    while (modelFiles.length > 0) {
      const file = modelFiles.shift();
      const filepath = path.normalize(modelsDir + '/' + file);
      
      fs.readFile(filepath, async (err, content) => {
        if (err) return console.log(err);
        
        const form = new FormData();
        form.append('file', content, { filepath, contentType: 'application/model' });
        try {
          const response = await axios.post(serverUrl, form, { headers: form.getHeaders() });
          if (response.status === 201) console.log(`success: ${file}`);
        } catch (err) {
          if (err.response) console.log('fail:', err.response.data);
          else console.log('fail:', err.message);
        }
      });
      
      await wait(1000*1); // make requests each 1s
    }
  });
};

const id = setInterval(backup, 1000*60*15); // do backup each 15mins

backup();
