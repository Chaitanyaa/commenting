$(document).ready(function () {
    // Init
    
    $('#result').hide();
    
    $('#btn-predict').show();

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                console.log('Success!');
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    });
    
    $('#btn-post').click(function () {
        var form_data = new FormData($('#comment')[0]);
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: false,
            success: function (data) {
                console.log('Success!');
                $('#res').val(data)
                console.log('Success!');
            },
        });     
        $("#comment").submit();
    });
});




